from pathlib import Path

import pandas as pd

from kdfp.utils.plot_gen import average_across_seeds


def test_average_across_seeds_uses_saved_experiment_directory_name(tmp_path: Path):
    students_dir = tmp_path / "models" / "students"
    experiment_name = "pretrained-kd-down_randomized_per-component-with_resid"
    (students_dir / experiment_name).mkdir(parents=True)

    df = pd.DataFrame(
        [
            {
                "EXPERIMENT": f"student_{experiment_name}_fixed_a1b2c3",
                "SEED": 3,
                "TOTAL_PARAMS": "270M",
                "TRAINABLE_PARAMS": "270M",
                "TASK": "hellaswag",
                "METRIC": "acc",
                "NUM_FEWSHOT": 0,
                "SCORE": 0.4,
            },
            {
                "EXPERIMENT": f"student_{experiment_name}_ray_trial_7f91",
                "SEED": 33,
                "TOTAL_PARAMS": "270M",
                "TRAINABLE_PARAMS": "270M",
                "TASK": "hellaswag",
                "METRIC": "acc",
                "NUM_FEWSHOT": 5,
                "SCORE": 0.6,
            },
        ]
    )

    averaged = average_across_seeds(df, model_students_dir=students_dir)

    assert len(averaged) == 1
    row = averaged.iloc[0]
    assert row["EXPERIMENT"] == experiment_name
    assert row["SCORE"] == 0.5
    assert row["N_SEEDS"] == 2


def test_average_across_seeds_preserves_unknown_experiment_names():
    df = pd.DataFrame(
        [
            {
                "EXPERIMENT": "baseline",
                "SEED": 3,
                "TASK": "hellaswag",
                "METRIC": "acc",
                "SCORE": 0.4,
            }
        ]
    )

    averaged = average_across_seeds(df)

    assert averaged.iloc[0]["EXPERIMENT"] == "baseline"
