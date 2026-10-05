#!/usr/bin/env python3
"""Compute pooled CKA and RSA between saved KDFP students and their teacher."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import re
import shutil
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from datasets import load_dataset
from scipy.stats import rankdata
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer


BLOCK_RE = re.compile(r"blocks\.(\d+)")
SEED_RE = re.compile(r"^seed\d+$")
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c",
        "--config",
        default="config.yaml",
        type=Path,
        help="Analysis config path; defaults to config.yaml.",
    )
    return parser.parse_args()


def resolve_config_path(config_path: Path) -> Path:
    """Prefer configs in this tool's directory, then use normal path resolution."""
    if not config_path.is_absolute():
        bundled_path = Path(__file__).resolve().parent / config_path
        if bundled_path.exists():
            return bundled_path
    return config_path


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("The analysis config must contain a YAML mapping.")
    if config.get("seq_len", 1024) != 1024:
        raise ValueError("seq_len must be exactly 1024 for this analysis.")
    config["seq_len"] = 1024
    config.setdefault("samples", 1000)
    config.setdefault("batch_size", 4)
    config.setdefault("device", "cuda" if torch.cuda.is_available() else "cpu")
    config.setdefault("dtype", "bfloat16")
    config.setdefault("seed", 67)
    if "analysis_seeds" in config and "seeds" in config:
        raise ValueError("Use only one of analysis_seeds or seeds.")
    analysis_seeds = config.get("analysis_seeds", config.get("seeds", [config["seed"]]))
    if not isinstance(analysis_seeds, list) or not analysis_seeds:
        raise ValueError("analysis_seeds must be a non-empty list of integers.")
    if not all(isinstance(seed, int) and not isinstance(seed, bool) for seed in analysis_seeds):
        raise ValueError("analysis_seeds must be a non-empty list of integers.")
    if len(set(analysis_seeds)) != len(analysis_seeds):
        raise ValueError("analysis_seeds must not contain duplicates.")
    config["analysis_seeds"] = analysis_seeds
    config.setdefault("activation_component", "resid")
    config.setdefault("metrics", ["cka", "rsa"])
    logging_config = config.setdefault("logging", {})
    if not isinstance(logging_config, dict):
        raise ValueError("logging must be a YAML mapping.")
    logging_config.setdefault("level", "INFO")
    logging_config.setdefault("progress", True)
    if not isinstance(logging_config["level"], str):
        raise ValueError("logging.level must be a string.")
    logging_level = logging_config["level"].upper()
    if logging_level not in logging._nameToLevel:
        raise ValueError(f"Unsupported logging.level: {logging_config['level']}")
    logging_config["level"] = logging_level
    if not isinstance(logging_config["progress"], bool):
        raise ValueError("logging.progress must be a boolean.")
    unknown_metrics = set(config["metrics"]) - {"cka", "rsa"}
    if unknown_metrics:
        raise ValueError(f"Unsupported metrics: {sorted(unknown_metrics)}")
    pooling = config.setdefault("pooling", {})
    pooling.setdefault("teacher_bins", 4)
    pooling.setdefault("student_bins", 4)
    if pooling["teacher_bins"] != 4 or pooling["student_bins"] != 4:
        raise ValueError("Both teacher_bins and student_bins must be exactly 4.")
    if not config.get("inputs"):
        raise ValueError("The config must provide at least one input directory.")
    normalized_inputs: list[dict[str, str]] = []
    seen_labels: set[str] = set()
    seen_output_labels: set[str] = set()
    for index, input_spec in enumerate(config["inputs"]):
        if isinstance(input_spec, str):
            input_path = input_spec
            label = Path(input_path).name
        elif isinstance(input_spec, dict):
            input_path = input_spec.get("path")
            label = input_spec.get("label")
            if not isinstance(input_path, str) or not input_path:
                raise ValueError(f"inputs[{index}].path must be a non-empty string.")
            if not isinstance(label, str) or not label.strip():
                raise ValueError(f"inputs[{index}].label must be a non-empty string.")
        else:
            raise ValueError(f"inputs[{index}] must be a path string or mapping with path and label.")
        output_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
        if not output_label:
            raise ValueError(f"inputs[{index}].label produces an empty output directory name.")
        if label in seen_labels:
            raise ValueError(f"Duplicate input label: {label}")
        if output_label in seen_output_labels:
            raise ValueError(f"Input labels collide after sanitization: {label}")
        seen_labels.add(label)
        seen_output_labels.add(output_label)
        normalized_inputs.append({"path": input_path, "label": label})
    config["inputs"] = normalized_inputs
    return config


def dtype_from_name(name: str) -> torch.dtype:
    try:
        return getattr(torch, name)
    except AttributeError as exc:
        raise ValueError(f"Unsupported torch dtype: {name}") from exc


def configure_reproducibility(seed: int, device: str) -> None:
    """Seed all relevant RNGs and enforce deterministic CUDA execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.use_deterministic_algorithms(True)


def reproducibility_metadata(seeds: int | list[int], device: str, dtype: str) -> dict[str, Any]:
    analysis_seeds = [seeds] if isinstance(seeds, int) else seeds
    metadata: dict[str, Any] = {
        "seed": analysis_seeds[0] if len(analysis_seeds) == 1 else None,
        "analysis_seeds": analysis_seeds,
        "device": device,
        "dtype": dtype,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    if torch.cuda.is_available():
        metadata["cuda_version"] = torch.version.cuda
        metadata["gpu"] = torch.cuda.get_device_name(0)
    return metadata


def seed_dirs(input_path: Path) -> list[Path]:
    if (input_path / "pl_config.json").exists():
        return [input_path]
    paths = sorted(
        path
        for path in input_path.iterdir()
        if path.is_dir() and SEED_RE.match(path.name) and (path / "pl_config.json").exists()
    )
    if not paths:
        raise FileNotFoundError(f"No seed directories containing pl_config.json: {input_path}")
    return paths


def read_student_config(path: Path) -> dict[str, Any]:
    with (path / "pl_config.json").open() as handle:
        return json.load(handle)


def component_hooks(hooks: list[str], component: str) -> list[str]:
    names = {
        "resid": "hook_resid_post",
        "attn_out": "hook_attn_out",
        "mlp": "hook_mlp_out",
    }
    if component not in names:
        raise ValueError(f"Unsupported activation_component: {component}")
    suffix = names[component]
    selected = [hook for hook in hooks if hook.endswith(suffix)]
    if not selected:
        raise ValueError(f"No {suffix} hooks found in pl_config.json")
    return selected


def layer_label(hook: str) -> str:
    match = BLOCK_RE.search(hook)
    return match.group(1) if match else hook


def load_model(
    model_name: str,
    device: str,
    dtype: torch.dtype,
    weights_path: Path | None = None,
) -> HookedTransformer:
    model = HookedTransformer.from_pretrained_no_processing(
        model_name,
        dtype=dtype,
        device=device,
    )
    if weights_path is not None:
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    model.eval()
    return model


def format_examples(dataset: Any, format_name: str) -> list[str]:
    if format_name == "question":
        return [str(row["question"]) for row in dataset]
    if format_name == "question_and_answer":
        return [f"Question: {row['question']}\nAnswer: {row['answer']}" for row in dataset]
    raise ValueError(f"Unsupported dataset.format: {format_name}")


def load_token_batches(
    texts: list[str], tokenizer: Any, batch_size: int, seed: int, device: str
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(
        texts,
        max_length=1024,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(texts), generator=generator)
    return [
        (
            encoded["input_ids"][order[start : start + batch_size]].to(device),
            encoded["attention_mask"][order[start : start + batch_size]].to(device),
        )
        for start in range(0, len(texts), batch_size)
    ]


def capture_activations(
    model: HookedTransformer,
    token_batches: list[tuple[torch.Tensor, torch.Tensor]],
    hooks: list[str],
    progress: bool = True,
    progress_desc: str = "Activation batches",
) -> dict[str, torch.Tensor]:
    collected = {hook: [] for hook in hooks}
    with torch.inference_mode():
        batches = tqdm(
            token_batches,
            desc=progress_desc,
            unit="batch",
            disable=not progress,
        )
        for tokens, attention_mask in batches:
            mask = attention_mask.bool()
            last_positions = mask.long().sum(dim=1).sub(1).clamp_min(0)
            _, cache = model.run_with_cache(tokens, names_filter=hooks)
            for hook in hooks:
                values = cache[hook]
                positions = last_positions.to(values.device)
                values = values[torch.arange(values.shape[0], device=values.device), positions]
                collected[hook].append(values.float().cpu())
    return {hook: torch.cat(values) for hook, values in collected.items()}


def cka(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    cross = x.T @ y
    xx = x.T @ x
    yy = y.T @ y
    denominator = torch.linalg.norm(xx) * torch.linalg.norm(yy)
    if denominator == 0:
        return float("nan")
    return float((torch.linalg.norm(cross) ** 2 / denominator).item())


def cosine_similarity_matrix(values: torch.Tensor) -> np.ndarray:
    values = torch.nn.functional.normalize(values, dim=1)
    return (values @ values.T).numpy()


def rsa(x: torch.Tensor, y: torch.Tensor) -> float:
    x_sim = cosine_similarity_matrix(x)
    y_sim = cosine_similarity_matrix(y)
    upper = np.triu_indices(x_sim.shape[0], k=1)
    x_rank = rankdata(x_sim[upper], method="average")
    y_rank = rankdata(y_sim[upper], method="average")
    x_rank -= x_rank.mean()
    y_rank -= y_rank.mean()
    denominator = np.linalg.norm(x_rank) * np.linalg.norm(y_rank)
    if denominator == 0:
        return float("nan")
    return float(np.dot(x_rank, y_rank) / denominator)


def similarity_matrix(
    teacher_acts: dict[str, torch.Tensor],
    student_acts: dict[str, torch.Tensor],
    teacher_hooks: list[str],
    student_hooks: list[str],
    metric: str,
    progress: bool = True,
) -> np.ndarray:
    matrix = np.empty((len(teacher_hooks), len(student_hooks)), dtype=np.float64)
    metric_fn = cka if metric == "cka" else rsa
    pairs = tqdm(
        total=len(teacher_hooks) * len(student_hooks),
        desc=f"{metric.upper()} layer pairs",
        unit="pair",
        disable=not progress,
    )
    for i, teacher_hook in enumerate(teacher_hooks):
        for j, student_hook in enumerate(student_hooks):
            matrix[i, j] = metric_fn(teacher_acts[teacher_hook], student_acts[student_hook])
            pairs.update(1)
    pairs.close()
    return matrix


def pooled_matrix(matrix: np.ndarray, teacher_bins: int = 4, student_bins: int = 4) -> np.ndarray:
    if matrix.shape[0] < teacher_bins or matrix.shape[1] < student_bins:
        raise ValueError("Each model must have at least four layers for a 4x4 pool.")
    pooled = np.empty((teacher_bins, student_bins), dtype=np.float64)
    teacher_groups = np.array_split(np.arange(matrix.shape[0]), teacher_bins)
    student_groups = np.array_split(np.arange(matrix.shape[1]), student_bins)
    for i, teacher_group in enumerate(teacher_groups):
        for j, student_group in enumerate(student_groups):
            pooled[i, j] = np.nanmean(matrix[np.ix_(teacher_group, student_group)])
    return pooled


def pooled_labels(hooks: list[str]) -> list[str]:
    if len(hooks) < 4:
        raise ValueError("Each model must have at least four layers for a 4x4 pool.")
    return [
        f"{layer_label(group[0])}-{layer_label(group[-1])}"
        for group in np.array_split(np.array(hooks), 4)
    ]


def write_csv(path: Path, matrix: np.ndarray, row_labels: list[str], column_labels: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["teacher\\student", *column_labels])
        writer.writerows([row_label, *row] for row_label, row in zip(row_labels, matrix))


def _vector_summary(values: np.ndarray, labels: list[str]) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"mean": None, "median": None, "minimum": None, "maximum": None, "range": None, "best_match": None, "best_similarity": None}
    best_index = int(np.nanargmax(values))
    minimum = float(np.min(finite))
    maximum = float(np.max(finite))
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "minimum": minimum,
        "maximum": maximum,
        "range": maximum - minimum,
        "best_match": labels[best_index],
        "best_similarity": float(values[best_index]),
    }


def _numeric_summary(values: list[float]) -> dict[str, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "mean": None,
            "median": None,
            "std": None,
            "minimum": None,
            "maximum": None,
            "range": None,
            "p10": None,
            "p25": None,
        }
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite)),
        "minimum": float(np.min(finite)),
        "maximum": float(np.max(finite)),
        "range": float(np.max(finite) - np.min(finite)),
        "p10": float(np.percentile(finite, 10)),
        "p25": float(np.percentile(finite, 25)),
    }


def matrix_summary(
    matrix: np.ndarray,
    teacher_labels: list[str],
    student_labels: list[str],
    thresholds: tuple[float, ...] = (0.25, 0.5, 0.75),
) -> dict[str, Any]:
    """Summarize a full layerwise matrix without computing another metric."""
    finite = matrix[np.isfinite(matrix)]
    summary: dict[str, Any] = {
        "shape": list(matrix.shape),
        "teacher_layers": teacher_labels,
        "student_layers": student_labels,
        "global": {
            "mean": float(np.mean(finite)) if finite.size else None,
            "median": float(np.median(finite)) if finite.size else None,
            "std": float(np.std(finite)) if finite.size else None,
            "minimum": float(np.min(finite)) if finite.size else None,
            "maximum": float(np.max(finite)) if finite.size else None,
            "range": float(np.max(finite) - np.min(finite)) if finite.size else None,
            "p10": float(np.percentile(finite, 10)) if finite.size else None,
            "p25": float(np.percentile(finite, 25)) if finite.size else None,
            "finite_pairs": int(finite.size),
            "total_pairs": int(matrix.size),
            "fraction_at_or_above": {
                str(threshold): float(np.mean(finite >= threshold)) if finite.size else None
                for threshold in thresholds
            },
        },
        "by_teacher_layer": {
            label: _vector_summary(matrix[index, :], student_labels)
            for index, label in enumerate(teacher_labels)
        },
        "by_student_layer": {
            label: _vector_summary(matrix[:, index], teacher_labels)
            for index, label in enumerate(student_labels)
        },
        "mapped_alignment": [],
    }
    if len(teacher_labels) != len(student_labels):
        raise ValueError(
            "Teacher and student hook lists must have the same length for mapped alignment."
        )
    for index, (teacher_label, student_label) in enumerate(zip(teacher_labels, student_labels)):
        summary["mapped_alignment"].append(
            {
                "teacher_layer": teacher_label,
                "student_layer": student_label,
                "similarity": (
                    float(matrix[index, index])
                    if np.isfinite(matrix[index, index])
                    else None
                ),
            }
        )
    student_best = [
        summary["by_student_layer"][label]["best_similarity"]
        for label in student_labels
    ]
    teacher_best = [
        summary["by_teacher_layer"][label]["best_similarity"]
        for label in teacher_labels
    ]
    depth_values = [
        entry["similarity"] for entry in summary["mapped_alignment"]
    ]
    summary["student_best_match"] = _numeric_summary(
        [value for value in student_best if value is not None]
    )
    summary["teacher_best_match"] = _numeric_summary(
        [value for value in teacher_best if value is not None]
    )
    summary["mapped_alignment_summary"] = _numeric_summary(
        [value for value in depth_values if value is not None]
    )
    return summary


def summary_scalars(summary: dict[str, Any]) -> dict[str, float | None]:
    """Extract comparable scalar values from a detailed matrix summary."""
    scalars = {
        "global_mean": summary["global"]["mean"],
        "global_median": summary["global"]["median"],
        "global_std": summary["global"]["std"],
        "global_p10": summary["global"]["p10"],
        "global_p25": summary["global"]["p25"],
        "global_minimum": summary["global"]["minimum"],
        "global_maximum": summary["global"]["maximum"],
        "global_range": summary["global"]["range"],
    }
    scalars.update(
        {
            f"global_fraction_at_or_above_{threshold.replace('.', '_')}": value
            for threshold, value in summary["global"]["fraction_at_or_above"].items()
        }
    )
    for prefix, section in (
        ("student_best", summary["student_best_match"]),
        ("teacher_best", summary["teacher_best_match"]),
        ("mapped", summary["mapped_alignment_summary"]),
    ):
        scalars.update(
            {
                f"{prefix}_{key}": section[key]
                for key in ("mean", "median", "std", "minimum", "maximum", "range", "p10", "p25")
            }
        )
    return scalars


def aggregate_seed_summaries(summaries: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    """Aggregate comparable summary scalars across trained seeds."""
    values = [summary_scalars(summary) for summary in summaries]
    return {
        key: _numeric_summary(
            [entry[key] for entry in values if entry[key] is not None]
        )
        for key in values[0]
    }


def plot_heatmap(path: Path, matrix: np.ndarray, title: str, row_labels: list[str], column_labels: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="viridis", vmin=-1, vmax=1, origin="lower")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color="white")
    ax.set_xticks(range(4), column_labels, rotation=45, ha="right")
    ax.set_yticks(range(4), row_labels)
    ax.set_xlabel("Student layer pool")
    ax.set_ylabel("Teacher layer pool")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="Similarity")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def load_gsm8k(config: dict[str, Any], seed: int) -> list[str]:
    dataset_config = config.get("dataset", {})
    dataset = load_dataset(
        dataset_config.get("name", "openai/gsm8k"),
        dataset_config.get("subset", "main"),
        split=dataset_config.get("split", "train"),
    )
    dataset = dataset.shuffle(seed=seed).select(range(min(config["samples"], len(dataset))))
    return format_examples(dataset, dataset_config.get("format", "question_and_answer"))


def analyze_input(
    input_path: Path,
    input_label: str,
    config: dict[str, Any],
    texts: list[str],
    teacher_cache: dict[tuple[str, str, str], tuple[HookedTransformer, Any]],
    analysis_seed: int,
) -> dict[str, Any]:
    progress = config["logging"]["progress"]
    paths = seed_dirs(input_path)
    seed_configs = [read_student_config(path) for path in paths]
    first = seed_configs[0]
    teacher_cfg = first["distill_pair"]["teacher"]
    for seed_cfg in seed_configs[1:]:
        if seed_cfg["distill_pair"]["teacher"] != teacher_cfg:
            raise ValueError(f"Teacher configuration differs across seeds in {input_path}")
    if first.get("finetune_teacher", False):
        raise NotImplementedError("finetune_teacher=true is not supported by this analysis yet.")

    dtype = dtype_from_name(config["dtype"])
    device = config["device"]
    component = config["activation_component"]
    teacher_hooks = component_hooks(teacher_cfg["hook_list"], component)
    student_hooks = component_hooks(first["distill_pair"]["student"]["hook_list"], component)
    teacher_key = (teacher_cfg["name"], config["device"], config["dtype"])
    if teacher_key not in teacher_cache:
        LOGGER.info("Loading teacher model %s", teacher_cfg["name"])
        teacher = load_model(teacher_cfg["name"], device, dtype)
        teacher_tokenizer = AutoTokenizer.from_pretrained(
            teacher_cfg.get("tokenizer_name") or teacher_cfg["name"]
        )
        teacher_cache[teacher_key] = (teacher, teacher_tokenizer)
    teacher, teacher_tokenizer = teacher_cache[teacher_key]
    LOGGER.info("Capturing teacher activations for %s", input_path)
    teacher_batches = load_token_batches(texts, teacher_tokenizer, config["batch_size"], analysis_seed, device)
    teacher_acts = capture_activations(
        teacher,
        teacher_batches,
        teacher_hooks,
        progress=progress,
        progress_desc="Teacher activations",
    )

    student_cfg = first["distill_pair"]["student"]
    LOGGER.info("Loading base student model %s", student_cfg["name"])
    base_student = load_model(student_cfg["name"], device, dtype)
    base_tokenizer = AutoTokenizer.from_pretrained(
        student_cfg.get("tokenizer_name") or student_cfg["name"]
    )
    base_batches = load_token_batches(texts, base_tokenizer, config["batch_size"], analysis_seed, device)
    base_acts = capture_activations(
        base_student,
        base_batches,
        student_hooks,
        progress=progress,
        progress_desc="Base student activations",
    )
    baseline: dict[str, dict[str, np.ndarray]] = {"matrices": {}, "full_matrices": {}}
    for metric in config["metrics"]:
        base_full = similarity_matrix(
            teacher_acts,
            base_acts,
            teacher_hooks,
            student_hooks,
            metric,
            progress=progress,
        )
        baseline["full_matrices"][metric] = base_full
        baseline["matrices"][metric] = pooled_matrix(base_full)
    del base_student
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    per_seed: dict[str, dict[str, np.ndarray]] = {"cka": {}, "rsa": {}}
    per_seed_full: dict[str, dict[str, np.ndarray]] = {"cka": {}, "rsa": {}}
    for path, seed_cfg in zip(paths, seed_configs):
        student_cfg = seed_cfg["distill_pair"]["student"]
        LOGGER.info("Loading trained student %s", path)
        student = load_model(
            student_cfg["name"],
            device,
            dtype,
            path / "student.bin",
        )
        student_tokenizer = AutoTokenizer.from_pretrained(path)
        student_batches = load_token_batches(texts, student_tokenizer, config["batch_size"], analysis_seed, device)
        student_acts = capture_activations(
            student,
            student_batches,
            student_hooks,
            progress=progress,
            progress_desc=f"{path.name} activations",
        )
        for metric in config["metrics"]:
            full = similarity_matrix(
                teacher_acts,
                student_acts,
                teacher_hooks,
                student_hooks,
                metric,
                progress=progress,
            )
            per_seed_full[metric][path.name] = full
            per_seed[metric][path.name] = pooled_matrix(full)
        del student
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    result: dict[str, Any] = {
        "input": str(input_path),
        "label": input_label,
        "analysis_seed": analysis_seed,
        "seeds": [path.name for path in paths],
        "teacher_model": teacher_cfg["name"],
        "student_model": first["distill_pair"]["student"]["name"],
        "teacher_hooks": teacher_hooks,
        "student_hooks": student_hooks,
        "teacher_layer_pools": pooled_labels(teacher_hooks),
        "student_layer_pools": pooled_labels(student_hooks),
        "baseline": baseline,
        "matrices": {},
        "std": {},
        "full_matrices": {},
        "full_std": {},
    }
    for metric in config["metrics"]:
        seed_matrices = per_seed[metric]
        values = np.stack(list(seed_matrices.values()))
        result["matrices"][metric] = values.mean(axis=0)
        result["std"][metric] = values.std(axis=0)
        result.setdefault("per_seed", {})[metric] = seed_matrices
        full_values = np.stack(list(per_seed_full[metric].values()))
        result["full_matrices"][metric] = full_values.mean(axis=0)
        result["full_std"][metric] = full_values.std(axis=0)
        result.setdefault("per_seed_full", {})[metric] = per_seed_full[metric]
    return result


def aggregate_analysis_results(results: list[dict[str, Any]], metrics: list[str]) -> dict[str, Any]:
    """Average one input's similarity matrices across analysis seeds."""
    if not results:
        raise ValueError("At least one analysis result is required.")
    first = results[0]
    if any(result["label"] != first["label"] for result in results[1:]):
        raise ValueError("Analysis-seed results have different input labels.")
    if any(result["seeds"] != first["seeds"] for result in results[1:]):
        raise ValueError("Analysis-seed results have different trained seed lists.")

    aggregated: dict[str, Any] = {
        key: first[key]
        for key in (
            "input",
            "label",
            "teacher_model",
            "student_model",
            "teacher_hooks",
            "student_hooks",
            "teacher_layer_pools",
            "student_layer_pools",
            "seeds",
        )
    }
    aggregated["analysis_seeds"] = [result["analysis_seed"] for result in results]
    aggregated["baseline"] = {"matrices": {}, "full_matrices": {}}
    aggregated["matrices"] = {}
    aggregated["std"] = {}
    aggregated["full_matrices"] = {}
    aggregated["full_std"] = {}
    aggregated["per_seed"] = {}
    aggregated["per_seed_full"] = {}

    for metric in metrics:
        baseline_full = np.mean(
            np.stack([result["baseline"]["full_matrices"][metric] for result in results]),
            axis=0,
        )
        aggregated["baseline"]["full_matrices"][metric] = baseline_full
        aggregated["baseline"]["matrices"][metric] = pooled_matrix(baseline_full)

        seed_names = first["per_seed_full"][metric].keys()
        aggregated_full_by_seed = {
            seed_name: np.mean(
                np.stack([result["per_seed_full"][metric][seed_name] for result in results]),
                axis=0,
            )
            for seed_name in seed_names
        }
        aggregated_pooled_by_seed = {
            seed_name: pooled_matrix(full_matrix)
            for seed_name, full_matrix in aggregated_full_by_seed.items()
        }
        aggregated["per_seed_full"][metric] = aggregated_full_by_seed
        aggregated["per_seed"][metric] = aggregated_pooled_by_seed
        seed_full_values = np.stack(list(aggregated_full_by_seed.values()))
        seed_pooled_values = np.stack(list(aggregated_pooled_by_seed.values()))
        aggregated["full_matrices"][metric] = seed_full_values.mean(axis=0)
        aggregated["full_std"][metric] = seed_full_values.std(axis=0)
        aggregated["matrices"][metric] = seed_pooled_values.mean(axis=0)
        aggregated["std"][metric] = seed_pooled_values.std(axis=0)
    return aggregated


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def write_metric_artifacts(
    directory: Path,
    metric: str,
    matrix: np.ndarray,
    full_matrix: np.ndarray,
    row_labels: list[str],
    column_labels: list[str],
    full_row_labels: list[str],
    full_column_labels: list[str],
    title: str,
    summary: dict[str, Any],
    std: np.ndarray | None = None,
    full_std: np.ndarray | None = None,
) -> dict[str, str]:
    metric_dir = directory / metric
    metric_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = metric_dir / "matrix.csv"
    full_matrix_path = metric_dir / "full_matrix.csv"
    heatmap_path = metric_dir / "heatmap.png"
    summary_path = metric_dir / "summary.json"
    write_csv(matrix_path, matrix, row_labels, column_labels)
    write_csv(full_matrix_path, full_matrix, full_row_labels, full_column_labels)
    plot_heatmap(heatmap_path, matrix, title, row_labels, column_labels)
    if std is not None:
        write_csv(metric_dir / "std.csv", std, row_labels, column_labels)
    if full_std is not None:
        write_csv(metric_dir / "full_std.csv", full_std, full_row_labels, full_column_labels)
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    paths = {
        "matrix": str(matrix_path),
        "full_matrix": str(full_matrix_path),
        "heatmap": str(heatmap_path),
        "summary": str(summary_path),
    }
    if std is not None:
        paths["std"] = str(metric_dir / "std.csv")
    if full_std is not None:
        paths["full_std"] = str(metric_dir / "full_std.csv")
    return paths


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(value, handle, indent=2)


def participant_summary(
    name: str,
    kind: str,
    result: dict[str, Any],
    metric: str,
    full_matrix: np.ndarray,
    seed_summaries: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    summary = matrix_summary(full_matrix, result["teacher_hooks"], result["student_hooks"])
    if seed_summaries is not None:
        summary["per_seed"] = seed_summaries
        summary["trained_seed_aggregate"] = aggregate_seed_summaries(list(seed_summaries.values()))
    return {
        "name": name,
        "kind": kind,
        "teacher_model": result["teacher_model"],
        "student_model": result["student_model"],
        "metric": metric,
        "summary": summary,
    }


def write_cross_student_summary(
    output_dir: Path,
    trained_records: list[dict[str, Any]],
    base_metrics: dict[str, dict[str, Any]],
    metrics: list[str],
) -> None:
    """Rank independent students by their similarity to the teacher."""
    summary_dir = output_dir / "summary"
    data_dir = summary_dir / "data"
    summary_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for metric in metrics:
        if metric in base_metrics:
            records.append(
                {
                    "model": "base_student",
                    "kind": "base_student",
                    "metric": metric,
                    "scores": summary_scalars(base_metrics[metric]),
                    "summary": str(output_dir / "base_student" / "summary" / "summary.json"),
                }
            )
        for trained in trained_records:
            records.append(
                {
                    "model": trained["name"],
                    "kind": "trained_student",
                    "metric": metric,
                    "scores": trained["metrics"][metric]["trained"],
                    "summary": trained["summary"],
                }
            )

    rankings: dict[str, list[dict[str, Any]]] = {}
    for metric in metrics:
        metric_records = [record for record in records if record["metric"] == metric]
        score_keys = sorted({key for record in metric_records for key in record["scores"]})
        for score_key in score_keys:
            lower_is_better = score_key.endswith("_std") or score_key.endswith("_range")
            ranked = [record for record in metric_records if record["scores"].get(score_key) is not None]
            ranked.sort(
                key=lambda record: record["scores"][score_key],
                reverse=not lower_is_better,
            )
            trained_ranked = [record for record in ranked if record["kind"] == "trained_student"]
            trained_ranks = {id(record): rank for rank, record in enumerate(trained_ranked, start=1)}
            ranking_name = f"{metric}_{score_key}"
            rankings[ranking_name] = [
                {
                    "rank_all": rank,
                    "rank_trained": trained_ranks.get(id(record)),
                    "model": record["model"],
                    "kind": record["kind"],
                    "score": record["scores"][score_key],
                    "direction": "lower_is_better" if lower_is_better else "higher_is_better",
                    "summary": record["summary"],
                }
                for rank, record in enumerate(ranked, start=1)
            ]

    for ranking_name, ranking in rankings.items():
        with (data_dir / f"{ranking_name}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["rank_all", "rank_trained", "model", "kind", "score", "direction", "summary"])
            writer.writerows(
                [entry["rank_all"], entry["rank_trained"], entry["model"], entry["kind"], entry["score"], entry["direction"], entry["summary"]]
                for entry in ranking
            )

    csv_score_keys = ("global_mean", "global_std", "global_p25", "mapped_mean", "mapped_std", "mapped_p25")
    score_keys_by_metric = {metric: list(csv_score_keys) for metric in metrics}
    student_rows: dict[str, dict[str, Any]] = {}
    rank_lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for ranking_name, ranking in rankings.items():
        metric, score_key = ranking_name.split("_", 1)
        for entry in ranking:
            rank_lookup[(entry["model"], metric, score_key)] = entry
    for record in records:
        row = student_rows.setdefault(record["model"], {"student": record["model"]})
        metric = record["metric"]
        for score_key in score_keys_by_metric[metric]:
            value = record["scores"].get(score_key)
            entry = rank_lookup[(record["model"], metric, score_key)]
            row[f"{metric}_{score_key}"] = value
            row[f"{metric}_{score_key}_rank"] = entry["rank_all"]
    csv_path = summary_dir / "student_metrics.csv"
    columns = ["student"]
    for metric in metrics:
        for score_key in score_keys_by_metric.get(metric, []):
            columns.extend([f"{metric}_{score_key}", f"{metric}_{score_key}_rank"])
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(student_rows.values())
    write_json(
        summary_dir / "summary.json",
        {
            "purpose": "Rank independent students by teacher-relative representational similarity.",
            "records": records,
            "rankings": rankings,
            "student_metrics_csv": str(csv_path),
        },
    )


def main() -> None:
    args = parse_args()
    config_path = resolve_config_path(args.config)
    config = load_config(config_path)
    configure_reproducibility(config["analysis_seeds"][0], config["device"])
    logging.basicConfig(
        level=logging._nameToLevel[config["logging"]["level"]],
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    output_dir = Path(config.get("output_dir", "representation_similarity_results"))
    LOGGER.info("Using config %s", config_path)
    LOGGER.info("Writing results to %s", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "config.yaml")

    teacher_cache: dict[tuple[str, str, str], tuple[HookedTransformer, Any]] = {}
    results_by_label: dict[str, list[dict[str, Any]]] = {
        input_spec["label"]: [] for input_spec in config["inputs"]
    }
    for analysis_seed in tqdm(
        config["analysis_seeds"],
        desc="Analysis seeds",
        unit="seed",
        disable=not config["logging"]["progress"],
    ):
        configure_reproducibility(analysis_seed, config["device"])
        LOGGER.info("Loading dataset for analysis seed %d", analysis_seed)
        texts = load_gsm8k(config, analysis_seed)
        LOGGER.info("Loaded %d examples for analysis seed %d", len(texts), analysis_seed)
        for input_spec in config["inputs"]:
            LOGGER.info("Analyzing input %s (%s), analysis seed %d", input_spec["label"], input_spec["path"], analysis_seed)
            results_by_label[input_spec["label"]].append(
                analyze_input(
                    Path(input_spec["path"]),
                    input_spec["label"],
                    config,
                    texts,
                    teacher_cache,
                    analysis_seed,
                )
            )
    results = [
        aggregate_analysis_results(input_results, config["metrics"])
        for input_results in results_by_label.values()
    ]

    metadata = {
        "analysis_seeds": config["analysis_seeds"],
        "samples": config["samples"],
        "seq_len": 1024,
        "activation_component": config["activation_component"],
        "reproducibility": reproducibility_metadata(
            config["analysis_seeds"], config["device"], config["dtype"]
        ),
        "models": [],
    }
    trained_records: list[dict[str, Any]] = []
    base_metrics: dict[str, dict[str, Any]] = {}
    base_written = False

    for result in results:
        model_name = safe_name(result["label"])
        model_dir = output_dir / model_name
        summary_dir = model_dir / "summary"
        seeds_dir = model_dir / "seeds"
        model_summary: dict[str, Any] = {
            "name": result["label"],
            "directory": model_name,
            "kind": "trained_student",
            "label": result["label"],
            "input": result["input"],
            "teacher_model": result["teacher_model"],
            "student_model": result["student_model"],
            "seeds": result["seeds"],
            "metrics": {},
            "base_student": {"metrics": {}},
            "comparisons": {},
            "files": {},
        }
        for metric in config["metrics"]:
            seed_summaries: dict[str, dict[str, Any]] = {}
            for seed_name, seed_matrix in result["per_seed_full"][metric].items():
                seed_dir = seeds_dir / safe_name(seed_name)
                seed_summary = matrix_summary(seed_matrix, result["teacher_hooks"], result["student_hooks"])
                seed_summaries[seed_name] = seed_summary
                seed_paths = write_metric_artifacts(
                    seed_dir,
                    metric,
                    result["per_seed"][metric][seed_name],
                    seed_matrix,
                    result["teacher_layer_pools"],
                    result["student_layer_pools"],
                    result["teacher_hooks"],
                    result["student_hooks"],
                    f"{metric.upper()} - {result['label']} {seed_name}",
                    seed_summary,
                )
                write_json(seed_dir / "summary.json", {"seed": seed_name, "metrics": {metric: seed_summary}, "files": {metric: seed_paths}})

            aggregate_summary = matrix_summary(
                result["full_matrices"][metric], result["teacher_hooks"], result["student_hooks"]
            )
            aggregate_summary["per_seed"] = seed_summaries
            aggregate_summary["trained_seed_aggregate"] = aggregate_seed_summaries(list(seed_summaries.values()))
            aggregate_paths = write_metric_artifacts(
                summary_dir,
                metric,
                result["matrices"][metric],
                result["full_matrices"][metric],
                result["teacher_layer_pools"],
                result["student_layer_pools"],
                result["teacher_hooks"],
                result["student_hooks"],
                f"{metric.upper()} - {result['label']}",
                aggregate_summary,
                result["std"][metric],
                result["full_std"][metric],
            )
            model_summary["metrics"][metric] = aggregate_summary
            model_summary["files"][metric] = aggregate_paths
            base_summary = matrix_summary(
                result["baseline"]["full_matrices"][metric],
                result["teacher_hooks"],
                result["student_hooks"],
            )
            base_dir = output_dir / "base_student"
            base_summary_dir = base_dir / "summary"
            base_paths = write_metric_artifacts(
                base_summary_dir,
                metric,
                result["baseline"]["matrices"][metric],
                result["baseline"]["full_matrices"][metric],
                result["teacher_layer_pools"],
                result["student_layer_pools"],
                result["teacher_hooks"],
                result["student_hooks"],
                f"{metric.upper()} - base student",
                base_summary,
            )
            model_summary["base_student"]["metrics"][metric] = base_summary
            base_metrics.setdefault(metric, base_summary)
            trained_scalars = summary_scalars(aggregate_summary)
            base_scalars = summary_scalars(base_summary)
            model_summary["comparisons"][metric] = {
                "trained": trained_scalars,
                "base": base_scalars,
                "trained_minus_base": {
                    key: (
                        trained_scalars[key] - base_scalars[key]
                        if trained_scalars[key] is not None and base_scalars[key] is not None
                        else None
                    )
                    for key in trained_scalars
                },
            }
            if not base_written:
                write_json(
                    base_summary_dir / "summary.json",
                    {"name": "base_student", "kind": "base_student", "metrics": {}, "files": {}},
                )
                base_written = True
            base_summary_document = json.loads((base_summary_dir / "summary.json").read_text())
            base_summary_document["metrics"][metric] = base_summary
            base_summary_document["files"][metric] = base_paths
            write_json(base_summary_dir / "summary.json", base_summary_document)
        write_json(summary_dir / "summary.json", model_summary)
        trained_records.append(
            {
                "name": result["label"],
                "directory": model_name,
                "summary": str(summary_dir / "summary.json"),
                "metrics": model_summary["comparisons"],
            }
        )
        metadata["models"].append({"name": result["label"], "directory": model_name, "path": str(model_dir), "summary": str(summary_dir / "summary.json")})

    write_cross_student_summary(output_dir, trained_records, base_metrics, config["metrics"])
    metadata["summary"] = {
        "directory": str(output_dir / "summary"),
        "summary": str(output_dir / "summary" / "summary.json"),
        "student_metrics": str(output_dir / "summary" / "student_metrics.csv"),
        "data": str(output_dir / "summary" / "data"),
    }
    write_json(output_dir / "metadata.json", metadata)
    LOGGER.info("Analysis complete; organized results saved to %s", output_dir)


if __name__ == "__main__":
    main()
