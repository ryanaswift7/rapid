import argparse
import gc
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from datasets import load_dataset
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from kdfp.utils.evals import evaluate_model
from kdfp.utils.general import is_valid_model_dir, set_all_seeds

# ==========================================
# Automated Dataset & Token Configurations
# ==========================================
DATASET_CONFIG = {
    "fineweb-edu": {
        "filter_dataset": "ryanaswift7/fineweb-edu-pythia-kdfp-subsets-100m",
        "student_dataset": "ryanaswift7/fineweb-edu-pythia-kdfp-subsets-100m",
        "tokens_s1": 25_000_000,
        "tokens_s2": 25_000_000,
        "tokens_s3": 100_000_000,
    },
    "fineweb-edu-500m": {
        "filter_dataset": "ryanaswift7/fineweb-edu-pythia-kdfp-subsets-500m",
        "student_dataset": "ryanaswift7/fineweb-edu-pythia-kdfp-subsets-500m",
        "tokens_s1": 125_000_000,
        "tokens_s2": 125_000_000,
        "tokens_s3": 500_000_000,
    }
}

EVAL_TASKS = [
    "arc_challenge", "arc_easy", "boolq", "triviaqa",
    "piqa", "sciq", "winogrande", "mmlu", "drop",
]

MODEL_CONFIGS = {
    "pythia": {
        "teacher": "EleutherAI/pythia-2.8b-deduped", "student": "EleutherAI/pythia-1.4b-deduped",
        "teacher_layers": 32, "student_layers": 24,
        "teacher_embedding_dim": 2560, "student_embedding_dim": 2048,
    },
    "pythia-bt": {
        "teacher": "EleutherAI/pythia-6.9b-deduped", "student": "EleutherAI/pythia-1.4b-deduped",
        "teacher_layers": 32, "student_layers": 24,
        "teacher_embedding_dim": 4096, "student_embedding_dim": 2048,
    },
    "pythia-ftt": {
        "teacher": "ryanaswift7/pythia-2.8b-deduped-ft-fwe-100m", "student": "EleutherAI/pythia-1.4b-deduped",
        "teacher_layers": 32, "student_layers": 24,
        "teacher_embedding_dim": 2560, "student_embedding_dim": 2048,
    },
    "llama3.2": {
        "teacher": "meta-llama/Llama-3.2-3B", "student": "meta-llama/Llama-3.2-1B",
        "teacher_layers": 28, "student_layers": 16,
        "teacher_embedding_dim": 3072, "student_embedding_dim": 2048,
    },
    "qwen2.5": {
        "teacher": "Qwen/Qwen2.5-3B", "student": "Qwen/Qwen2.5-1.5B",
        "teacher_layers": 36, "student_layers": 28,
        "teacher_embedding_dim": 2048, "student_embedding_dim": 1536,
    },
    "gemma3": {
        "teacher": "google/gemma-3-4b-pt", "student": "google/gemma-3-1b-pt",
        "teacher_layers": 34, "student_layers": 26,
        "teacher_embedding_dim": 2560, "student_embedding_dim": 2304,
    },
    "qwen3": {
        "teacher": "Qwen/Qwen3-4B", "student": "Qwen/Qwen3-1.7B",
        "teacher_layers": 36, "student_layers": 28,
        "teacher_embedding_dim": 2560, "student_embedding_dim": 2048,
    }
}

# [TaskAwareFilter, get_norm_and_head, TEDDistiller, run_training_loop classes/functions remain identical to your provided code]

class TaskAwareFilter(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = output_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, hidden_state):
        return self.mlp(hidden_state)

def get_norm_and_head(model):
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm, model.lm_head
    elif hasattr(model, "gpt_neox"):
        return model.gpt_neox.final_layer_norm, model.embed_out
    else:
        raise ValueError("Unsupported model architecture for extracting norm/head.")

class TEDDistiller(nn.Module):
    def __init__(self, teacher, student, teacher_filters, student_filters, layer_map):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.teacher_filters = teacher_filters
        self.student_filters = student_filters
        self.layer_map = layer_map

    def forward(self, input_ids):
        with torch.no_grad():
            t_outputs = self.teacher(input_ids, output_hidden_states=True, return_dict=True, use_cache=False)
            t_logits = t_outputs.logits
            t_hiddens = {int(l): t_outputs.hidden_states[int(l) + 1] for l in self.layer_map}

        s_outputs = self.student(input_ids, labels=input_ids, output_hidden_states=True, return_dict=True, use_cache=False)
        s_logits = s_outputs.logits
        ce_loss = s_outputs.loss
        s_hiddens = [s_outputs.hidden_states[l + 1] for l in range(len(self.student_filters))]

        s_vocab_size = s_logits.size(-1)
        t_vocab_size = t_logits.size(-1)

        if t_vocab_size > s_vocab_size:
            t_logits = t_logits[..., :s_vocab_size]
        elif s_vocab_size > t_vocab_size:
            padding = torch.full((*s_logits.shape[:-1], s_vocab_size - t_vocab_size), -1e4, device=t_logits.device, dtype=t_logits.dtype)
            t_logits = torch.cat([t_logits, padding], dim=-1)

        T = 2.0
        t_probs = F.softmax(t_logits / T, dim=-1)
        kl_loss = F.kl_div(F.log_softmax(s_logits / T, dim=-1), t_probs, reduction="batchmean") * (T**2)

        layer_loss = 0.0
        mse = nn.MSELoss()
        for s_idx, t_idx in enumerate(self.layer_map):
            with torch.no_grad():
                t_target = self.teacher_filters[str(t_idx)](t_hiddens[int(t_idx)])
            s_mapped = self.student_filters[s_idx](s_hiddens[s_idx])
            layer_loss += mse(s_mapped, t_target)
        layer_loss = layer_loss / len(self.layer_map)

        return ce_loss + (0.1 * layer_loss) + (2.5 * kl_loss)

def run_training_loop(stage_name, target_tokens, data_iterator, loss_fn, optimizer, device):
    tokens_processed = 0
    last_loss = 0.0
    step_count = 0
    pbar = tqdm(total=target_tokens, desc=f"[{stage_name}]", unit="tok", leave=True)

    while tokens_processed < target_tokens:
        try:
            batch = next(data_iterator)
        except StopIteration:
            break

        input_ids = batch["input_ids"].to(device)
        current_tokens = input_ids.numel()
        loss = loss_fn(input_ids)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tokens_processed += current_tokens
        last_loss = loss.item()
        pbar.update(current_tokens)
        pbar.set_postfix({"loss": f"{last_loss:.4f}"})
        del loss, input_ids, batch
        
        step_count += 1
        if step_count % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    pbar.close()
    return last_loss

# Stage functions (Warmups/Distill) remain logic-identical
def teacher_warmup_stage(config, data_iterator, temp_dir, device, dtype):
    model_cfg = MODEL_CONFIGS[config["model_family"]]
    teacher = AutoModelForCausalLM.from_pretrained(model_cfg["teacher"], torch_dtype=dtype).to(device)
    for p in teacher.parameters(): p.requires_grad = False
    teacher.eval()
    layer_map = np.linspace(0, model_cfg["teacher_layers"] - 1, model_cfg["student_layers"], dtype=int)
    teacher_filters = nn.ModuleDict({str(t_idx): TaskAwareFilter(model_cfg["teacher_embedding_dim"], model_cfg["teacher_embedding_dim"]) for t_idx in layer_map}).to(device, dtype=dtype)
    opt = torch.optim.AdamW(teacher_filters.parameters(), lr=config["lr_filters"], fused=True)
    norm, head = get_norm_and_head(teacher)

    def loss_fn(input_ids):
        with torch.no_grad():
            outputs = teacher(input_ids, output_hidden_states=True, return_dict=True, use_cache=False)
            t_hiddens = {int(l): outputs.hidden_states[int(l) + 1] for l in layer_map}
        loss = 0.0
        for t_idx in layer_map:
            t_filt = teacher_filters[str(t_idx)](t_hiddens[int(t_idx)])
            normed = norm(t_filt)
            logits = head(normed)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss += F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss / len(layer_map)

    last_loss = run_training_loop("teacher_warmup", config["tokens_s1"], data_iterator, loss_fn, opt, device)
    torch.save(teacher_filters.state_dict(), Path(temp_dir) / "teacher_filters.pt")
    del teacher, teacher_filters, opt, loss_fn
    gc.collect(); torch.cuda.empty_cache()
    return last_loss

def student_warmup_stage(config, data_iterator, temp_dir, device, dtype):
    model_cfg = MODEL_CONFIGS[config["model_family"]]
    teacher = AutoModelForCausalLM.from_pretrained(model_cfg["teacher"], torch_dtype=dtype).to(device)
    norm, head = get_norm_and_head(teacher)
    if hasattr(teacher, "model"): del teacher.model.layers
    elif hasattr(teacher, "gpt_neox"): del teacher.gpt_neox.layers
    gc.collect(); torch.cuda.empty_cache()
    for p in norm.parameters(): p.requires_grad = False
    for p in head.parameters(): p.requires_grad = False
    norm.eval(); head.eval()

    student = AutoModelForCausalLM.from_pretrained(model_cfg["student"], torch_dtype=dtype).to(device)
    student.gradient_checkpointing_enable()
    student_filters = nn.ModuleList([TaskAwareFilter(model_cfg["student_embedding_dim"], model_cfg["teacher_embedding_dim"]) for _ in range(model_cfg["student_layers"])]).to(device, dtype=dtype)
    opt = torch.optim.AdamW(student_filters.parameters(), lr=config["lr_filters"], fused=True)

    def loss_fn(input_ids):
        outputs = student(input_ids, output_hidden_states=True, return_dict=True, use_cache=False)
        s_hiddens = [outputs.hidden_states[l + 1] for l in range(model_cfg["student_layers"])]
        loss = 0.0
        for s_idx, h in enumerate(s_hiddens):
            s_filt = student_filters[s_idx](h); normed = norm(s_filt); logits = head(normed)
            shift_logits = logits[..., :-1, :].contiguous(); shift_labels = input_ids[..., 1:].contiguous()
            loss += F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss / len(student_filters)

    last_loss = run_training_loop("student_warmup", config["tokens_s2"], data_iterator, loss_fn, opt, device)
    torch.save(student_filters.state_dict(), Path(temp_dir) / "student_filters.pt")
    del student, student_filters, teacher, norm, head, opt, loss_fn
    gc.collect(); torch.cuda.empty_cache()
    return last_loss

def distillation_stage(config, data_iterator, temp_dir, cumulative_tokens, device, dtype):
    model_cfg = MODEL_CONFIGS[config["model_family"]]
    layer_map = np.linspace(0, model_cfg["teacher_layers"] - 1, model_cfg["student_layers"], dtype=int)
    teacher = AutoModelForCausalLM.from_pretrained(model_cfg["teacher"], torch_dtype=dtype).to(device)
    for p in teacher.parameters(): p.requires_grad = False
    teacher.eval()
    student = AutoModelForCausalLM.from_pretrained(model_cfg["student"], torch_dtype=dtype).to(device)
    student.gradient_checkpointing_enable()
    teacher_filters = nn.ModuleDict({str(t_idx): TaskAwareFilter(model_cfg["teacher_embedding_dim"], model_cfg["teacher_embedding_dim"]) for t_idx in layer_map}).to(device, dtype=dtype)
    teacher_filters.load_state_dict(torch.load(Path(temp_dir) / "teacher_filters.pt", weights_only=True))
    for p in teacher_filters.parameters(): p.requires_grad = False
    student_filters = nn.ModuleList([TaskAwareFilter(model_cfg["student_embedding_dim"], model_cfg["teacher_embedding_dim"]) for _ in range(model_cfg["student_layers"])]).to(device, dtype=dtype)
    student_filters.load_state_dict(torch.load(Path(temp_dir) / "student_filters.pt", weights_only=True))
    model = TEDDistiller(teacher, student, teacher_filters, student_filters, layer_map)
    params = list(student.parameters()) + list(student_filters.parameters())
    opt = torch.optim.AdamW(params, lr=config["lr_student"], fused=True)
    num_chunks = 10
    chunk_size = config["tokens_s3"] // num_chunks
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["student"])

    def loss_fn(input_ids): return model(input_ids)

    for i in range(num_chunks):
        loss_val = run_training_loop(f"distill_chunk_{i+1}/{num_chunks}", chunk_size, data_iterator, loss_fn, opt, device)
        cumulative_tokens += chunk_size
        if i == (num_chunks - 1):
            with tempfile.TemporaryDirectory() as temp_ckpt:
                student.save_pretrained(temp_ckpt); tokenizer.save_pretrained(temp_ckpt)
                train.report({"loss": loss_val, "tokens_processed": cumulative_tokens}, checkpoint=Checkpoint.from_directory(temp_ckpt))
        else:
            train.report({"loss": loss_val, "tokens_processed": cumulative_tokens})
    del model, teacher, student, teacher_filters, student_filters, opt, loss_fn
    gc.collect(); torch.cuda.empty_cache()

# ==========================================
# Main Ray Worker Function
# ==========================================
def train_ted_ray(config: Dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    model_cfg = MODEL_CONFIGS[config["model_family"]]
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["student"])
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    dataset_name = config["dataset"]
    
    # Check if we are using an automated Fineweb-Edu configuration
    if dataset_name in DATASET_CONFIG:
        cfg = DATASET_CONFIG[dataset_name]
        # For Stage 1/2 filters, we use the filter repo, then switch to student repo for Stage 3
        # However, to keep it simple with one DataLoader, we pull from the larger student repo which contains all subsets
        dataset = load_dataset(cfg["student_dataset"], "v1", split="train", streaming=True)
    elif dataset_name == "wikipedia": 
        dataset = load_dataset("ryanaswift7/wikipedia-pythia-kdfp-subsets-100m", "v1", split="train", streaming=True)
    elif dataset_name == "dolma3": 
        dataset = load_dataset("ryanaswift7/dolma3-pythia-kdfp-subsets-100m", "v1", split="train", streaming=True)
    else: 
        raise ValueError(f"Unknown dataset: {dataset_name}")

    def tokenize(ex):
        return tokenizer(ex["text"], truncation=True, padding="max_length", max_length=config["seq_len"])

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    dataset = dataset.with_format("torch")
    loader = DataLoader(dataset, batch_size=config["batch_size"])
    data_iterator = iter(loader)

    temp_filter_dir = tempfile.mkdtemp()
    try:
        teacher_warmup_stage(config, data_iterator, temp_filter_dir, device, dtype)
        s2_loss = student_warmup_stage(config, data_iterator, temp_filter_dir, device, dtype)
        cumulative_tokens = config["tokens_s1"] + config["tokens_s2"]
        train.report({"loss": s2_loss, "tokens_processed": cumulative_tokens})
        distillation_stage(config, data_iterator, temp_filter_dir, cumulative_tokens, device, dtype)
    finally:
        shutil.rmtree(temp_filter_dir, ignore_errors=True)

# [run_evaluation remains identical to your provided code]
def run_evaluation(model_path: str | Path, tasks: list, seed: int, eval_limit: int = None, device: str = "cuda"):
    model_path = Path(model_path)
    print(f"\n=== Running Eval on {model_path} ===")
    if not is_valid_model_dir(model_path):
        print(f"Error: No valid model found at {model_path}"); return
    out_dir = model_path / "scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "conventional_raw_eval_scores.json"
    extracted_path = out_dir / "conventional_extracted_eval_scores.json"
    csv_path = out_dir / "eval_scores.csv"
    parquet_path = out_dir / "eval_scores.parquet"

    if raw_path.exists() and extracted_path.exists():
        with open(raw_path, "r") as f:
            raw_results_dict = json.load(f)
        with open(extracted_path, "r") as f:
            extracted_dict = json.load(f)
        if parquet_path.exists():
            results_df = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            results_df = pd.read_csv(csv_path)
        else:
            results_df = pd.DataFrame()
        print(f"Loaded cached eval scores from {out_dir}")
    else:
        experiment_name = model_path.name
        extracted_dict, results_df, raw_results_dict = evaluate_model(
            model_path=str(model_path),
            tasks=tasks,
            experiment_name=experiment_name,
            batch_size="auto",
            limit=eval_limit,
            device=device,
            seed=seed)
        with open(raw_path, "w") as f: json.dump(raw_results_dict, f, indent=4)
        with open(extracted_path, "w") as f: json.dump(extracted_dict, f, indent=4)
        results_df.to_csv(csv_path, index=False)
        if not results_df.empty:
            results_df.to_parquet(parquet_path, index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--model-family", type=str, choices=["pythia", "pythia-bt", "pythia-ftt", "llama3.2", "qwen2.5", "gemma3", "qwen3"], required=True)
    parser.add_argument("--dataset", type=str, choices=["fineweb-edu", "fineweb-edu-500m", "wikipedia", "dolma3"], required=True)
    parser.add_argument("--lr-filters", type=float, default=None)
    parser.add_argument("--lr-student", type=float, default=None)
    parser.add_argument("--force-lr-tune", action="store_true")
    parser.add_argument("--tokens-s1", type=int, default=25_000_000)
    parser.add_argument("--tokens-s2", type=int, default=25_000_000)
    parser.add_argument("--tokens-s3", type=int, default=100_000_000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    set_all_seeds(seed=args.seed, reproducible_backend=True)

    # --- Apply Automated Budgeting ---
    tokens_s1, tokens_s2, tokens_s3 = args.tokens_s1, args.tokens_s2, args.tokens_s3
    if args.dataset in DATASET_CONFIG:
        auto_cfg = DATASET_CONFIG[args.dataset]
        tokens_s1 = auto_cfg["tokens_s1"]
        tokens_s2 = auto_cfg["tokens_s2"]
        tokens_s3 = auto_cfg["tokens_s3"]
        print(f"!!! Automated Budget Applied for {args.dataset} !!!")
        print(f"S1: {tokens_s1} | S2: {tokens_s2} | S3: {tokens_s3}")

    should_train = True
    if args.skip_train:
        should_train = False
    elif is_valid_model_dir(args.save_path):
        print(f"Model found at {args.save_path}. Skipping training.")
        should_train = False

    if should_train:
        ray.init()
        search_space = {
            "lr_filters": args.lr_filters if args.lr_filters is not None else tune.qloguniform(1e-4, 1e-2, 1e-5),
            "lr_student": args.lr_student if args.lr_student is not None else tune.qloguniform(1e-6, 1e-4, 1e-7),
            "tokens_s1": tokens_s1,
            "tokens_s2": tokens_s2,
            "tokens_s3": tokens_s3,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "model_family": args.model_family,
            "dataset": args.dataset,
        }

        total_tokens = tokens_s1 + tokens_s2 + tokens_s3
        grace_tokens = tokens_s1 + tokens_s2 + (tokens_s3 // 10) 

        scheduler = ASHAScheduler(
            time_attr="tokens_processed",
            max_t=total_tokens,
            grace_period=grace_tokens,
            reduction_factor=3, 
        )

        num_trials = 25 if (args.lr_filters is None or args.lr_student is None or args.force_lr_tune) else 1
        tuner = tune.Tuner(
            tune.with_resources(train_ted_ray, resources={"cpu": 4, "gpu": 1}),
            param_space=search_space,
            tune_config=tune.TuneConfig(metric="loss", mode="min", scheduler=scheduler, num_samples=num_trials),
            run_config=train.RunConfig(
                storage_path=os.path.abspath("./ray_results"), 
                name="ted_stream_sweep",
                checkpoint_config=train.CheckpointConfig(num_to_keep=1, checkpoint_score_attribute="loss", checkpoint_score_order="min")
            ),
        )

        results = tuner.fit()
        best_result = results.get_best_result("loss", "min")
        with best_result.checkpoint.as_directory() as checkpoint_dir:
            shutil.copytree(checkpoint_dir, args.save_path, dirs_exist_ok=True)
        
        with open(Path(args.save_path) / "final_learning_rates.json", "w") as f:
            json.dump({"lr_filters": best_result.config.get("lr_filters"), "lr_student": best_result.config.get("lr_student")}, f, indent=4)
        
        ray.shutdown(); torch.cuda.empty_cache(); gc.collect()

    run_evaluation(model_path=args.save_path, tasks=EVAL_TASKS, eval_limit=args.eval_limit, seed=args.seed)

if __name__ == "__main__":
    main()
