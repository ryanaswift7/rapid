import json

import numpy as np
import torch
from pytest import approx, raises

from representation_similarity.analyze import (
    aggregate_seed_summaries,
    aggregate_analysis_results,
    configure_reproducibility,
    load_config,
    load_model,
    matrix_summary,
    plot_heatmap,
    summary_scalars,
    reproducibility_metadata,
    write_cross_student_summary,
)


class FakeModel:
    def __init__(self):
        self.loaded_state = None
        self.eval_called = False

    def load_state_dict(self, state_dict):
        self.loaded_state = state_dict

    def eval(self):
        self.eval_called = True
        return self


def test_load_model_uses_hooked_transformer_for_trained_checkpoint(monkeypatch, tmp_path):
    model = FakeModel()
    calls = []

    def from_pretrained_no_processing(model_name, dtype, device):
        calls.append((model_name, dtype, device))
        return model

    monkeypatch.setattr(
        "representation_similarity.analyze.HookedTransformer.from_pretrained_no_processing",
        from_pretrained_no_processing,
    )
    weights_path = tmp_path / "student.bin"
    torch.save({"weight": torch.ones(1)}, weights_path)

    loaded = load_model("base-student", "cpu", torch.float32, weights_path)

    assert loaded is model
    assert calls == [("base-student", torch.float32, "cpu")]
    assert model.loaded_state["weight"].equal(torch.ones(1))
    assert model.eval_called


def test_matrix_summary_records_layer_matches_and_thresholds():
    summary = matrix_summary(
        np.array([[0.1, 0.8], [0.4, 0.6]]),
        ["teacher-0", "teacher-1"],
        ["student-0", "student-1"],
    )

    assert summary["global"]["mean"] == 0.475
    assert summary["global"]["std"] == approx(0.2586020108)
    assert summary["global"]["p10"] == approx(0.19)
    assert summary["global"]["p25"] == approx(0.325)
    assert summary["global"]["fraction_at_or_above"]["0.5"] == 0.5
    assert summary["by_teacher_layer"]["teacher-0"]["best_match"] == "student-1"
    assert summary["by_student_layer"]["student-0"]["best_match"] == "teacher-1"
    assert summary["student_best_match"]["mean"] == approx(0.6)
    assert summary["mapped_alignment_summary"]["mean"] == 0.35
    assert summary["mapped_alignment"] == [
        {"teacher_layer": "teacher-0", "student_layer": "student-0", "similarity": 0.1},
        {"teacher_layer": "teacher-1", "student_layer": "student-1", "similarity": 0.6},
    ]


def test_matrix_summary_rejects_unmatched_mapping_lengths():
    with raises(ValueError, match="same length for mapped alignment"):
        matrix_summary(np.zeros((2, 2)), ["teacher-0", "teacher-1"], ["student-0"])


def test_aggregate_seed_summaries_reports_mean_and_std():
    summaries = [
        matrix_summary(np.array([[0.2, 0.4], [0.6, 0.8]]), ["t0", "t1"], ["s0", "s1"]),
        matrix_summary(np.array([[0.4, 0.6], [0.8, 1.0]]), ["t0", "t1"], ["s0", "s1"]),
    ]

    aggregate = aggregate_seed_summaries(summaries)

    assert aggregate["global_mean"]["mean"] == 0.6
    assert aggregate["global_mean"]["std"] == approx(0.1)
    assert aggregate["student_best_mean"]["mean"] == 0.8


def test_load_config_reads_logging_settings(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "inputs: [example]\n"
        "seq_len: 1024\n"
        "logging:\n"
        "  level: debug\n"
        "  progress: false\n"
    )

    config = load_config(config_path)

    assert config["logging"] == {"level": "DEBUG", "progress": False}
    assert config["analysis_seeds"] == [67]


def test_configure_reproducibility_reseeds_cpu_rngs():
    configure_reproducibility(123, "cpu")
    first = (np.random.random(), torch.rand(1).item())
    configure_reproducibility(123, "cpu")
    second = (np.random.random(), torch.rand(1).item())

    assert first == second
    metadata = reproducibility_metadata(123, "cpu", "float32")
    assert metadata["seed"] == 123
    assert metadata["device"] == "cpu"


def test_aggregate_analysis_results_averages_analysis_seed_matrices():
    common = {
        "input": "experiment",
        "label": "student",
        "teacher_model": "teacher",
        "student_model": "student",
        "teacher_hooks": ["blocks.0.hook_resid_post"] * 4,
        "student_hooks": ["blocks.0.hook_resid_post"] * 4,
        "teacher_layer_pools": ["0-0", "1-1", "2-2", "3-3"],
        "student_layer_pools": ["0-0", "1-1", "2-2", "3-3"],
        "seeds": ["seed3"],
    }

    def result(seed, value):
        full = np.full((4, 4), value)
        pooled = np.full((4, 4), value)
        return {
            **common,
            "analysis_seed": seed,
            "baseline": {"matrices": {"cka": pooled}, "full_matrices": {"cka": full}},
            "matrices": {"cka": pooled},
            "std": {"cka": np.zeros((4, 4))},
            "full_matrices": {"cka": full},
            "full_std": {"cka": np.zeros((4, 4))},
            "per_seed": {"cka": {"seed3": pooled}},
            "per_seed_full": {"cka": {"seed3": full}},
        }

    aggregated = aggregate_analysis_results([result(67, 0.2), result(123, 0.8)], ["cka"])

    assert aggregated["analysis_seeds"] == [67, 123]
    assert np.allclose(aggregated["full_matrices"]["cka"], 0.5)
    assert np.allclose(aggregated["baseline"]["matrices"]["cka"], 0.5)


def test_load_config_normalizes_labeled_inputs(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "inputs:\n"
        "  - label: Student A / Online KD\n"
        "    path: experiments/a\n"
        "  - experiments/b\n"
        "seq_len: 1024\n"
    )

    config = load_config(config_path)

    assert config["inputs"] == [
        {"label": "Student A / Online KD", "path": "experiments/a"},
        {"label": "b", "path": "experiments/b"},
    ]


def test_plot_heatmap_uses_lower_origin_and_exact_title(monkeypatch, tmp_path):
    captured = {}

    class FakeAxis:
        def imshow(self, matrix, **kwargs):
            captured["origin"] = kwargs["origin"]

        def text(self, *args, **kwargs):
            pass

        def set_xticks(self, *args, **kwargs):
            pass

        def set_yticks(self, *args, **kwargs):
            pass

        def set_xlabel(self, *args, **kwargs):
            pass

        def set_ylabel(self, *args, **kwargs):
            pass

        def set_title(self, title):
            captured["title"] = title

    class FakeFigure:
        def colorbar(self, *args, **kwargs):
            pass

        def tight_layout(self):
            pass

        def savefig(self, *args, **kwargs):
            pass

    monkeypatch.setattr("representation_similarity.analyze.plt.subplots", lambda **kwargs: (FakeFigure(), FakeAxis()))
    monkeypatch.setattr("representation_similarity.analyze.plt.close", lambda figure: None)

    plot_heatmap(
        tmp_path / "heatmap.png",
        np.zeros((4, 4)),
        "CKA - Student A / Online KD",
        ["0-1", "2-3", "4-5", "6-7"],
        ["0-1", "2-3", "4-5", "6-7"],
    )

    assert captured == {"origin": "lower", "title": "CKA - Student A / Online KD"}


def test_cross_student_summary_ranks_teacher_relative_scores(tmp_path):
    teacher_hooks = [f"blocks.{index}.hook_resid_post" for index in range(4)]
    student_hooks = [f"blocks.{index}.hook_resid_post" for index in range(4)]
    base_summary = matrix_summary(
        np.full((4, 4), 0.4), teacher_hooks, student_hooks
    )
    better_summary = matrix_summary(
        np.full((4, 4), 0.8), teacher_hooks, student_hooks
    )
    worse_summary = matrix_summary(
        np.full((4, 4), 0.2), teacher_hooks, student_hooks
    )
    write_cross_student_summary(
        tmp_path,
        [
            {
                "name": "student_a",
                "summary": "student_a/summary/summary.json",
                "metrics": {"cka": {"trained": summary_scalars(better_summary)}},
            },
            {
                "name": "student_b",
                "summary": "student_b/summary/summary.json",
                "metrics": {"cka": {"trained": summary_scalars(worse_summary)}},
            },
        ],
        {"cka": base_summary},
        ["cka"],
    )

    ranking_path = tmp_path / "summary" / "data" / "cka_global_mean.csv"
    summary_path = tmp_path / "summary" / "summary.json"
    student_metrics_path = tmp_path / "summary" / "student_metrics.csv"
    assert ranking_path.exists()
    assert summary_path.exists()
    assert student_metrics_path.exists()
    assert not (tmp_path / "cross_student").exists()
    summary = json.loads(summary_path.read_text())
    assert "cka_global_maximum" in summary["rankings"]
    assert "cka_global_fraction_at_or_above_0_5" in summary["rankings"]
    assert "cka_student_best_std" in summary["rankings"]
    assert "cka_mapped_mean" in summary["rankings"]
    assert "cka_relative_depth_mean" not in summary["rankings"]
    assert summary["rankings"]["cka_student_best_std"][0]["direction"] == "lower_is_better"
    assert summary["rankings"]["cka_global_mean"][0]["model"] == "student_a"
    header, *rows = student_metrics_path.read_text().splitlines()
    assert header.startswith("student,cka_global_mean,cka_global_mean_rank")
    assert "cka_global_std" in header
    assert "cka_mapped_mean" in header
    assert "rsa_global_mean" not in header
    assert "cka_student_best_mean" not in header
    assert "kind" not in header
    assert "average_rank" not in header
    assert len(rows) == 3
