from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import pytest

import kdfp.schemas.experiments.experimentInstance as experiment_instance_module
from kdfp.schemas.experiments.experimentInstance import ExperimentInstance


def _instance_with_force_rerun() -> ExperimentInstance:
    return ExperimentInstance.model_construct(
        tasks=type("Tasks", (), {"force_rerun": True})()
    )


def test_force_rerun_archives_existing_scores(tmp_path: Path) -> None:
    scores_dir = tmp_path / "scores"
    scores_dir.mkdir()
    previous = scores_dir / "conventional_raw_eval_scores.json"
    previous.write_text("previous")

    _instance_with_force_rerun()._save_eval_scores(
        scores_dir=scores_dir,
        extracted_dict={"task": {"acc": 1.0}},
        results_df=pd.DataFrame([{"SCORE": 1.0}]),
        raw_results_dict={"task": {"acc": 1.0}},
    )

    archives = list(tmp_path.glob("scores-archive-*"))
    assert len(archives) == 1
    assert (archives[0] / previous.name).read_text() == "previous"
    assert (scores_dir / previous.name).read_text() != "previous"


def test_force_rerun_uses_a_new_archive_name_on_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FixedDateTime:
        @classmethod
        def now(cls, tz=None) -> datetime:
            return datetime(2026, 9, 23, tzinfo=timezone.utc)

    monkeypatch.setattr(experiment_instance_module, "datetime", FixedDateTime)

    scores_dir = tmp_path / "scores"
    scores_dir.mkdir()
    (scores_dir / "old.txt").write_text("old")

    instance = _instance_with_force_rerun()
    instance._save_eval_scores(
        scores_dir=scores_dir,
        extracted_dict={},
        results_df=pd.DataFrame(),
        raw_results_dict={},
    )

    first_archive = next(tmp_path.glob("scores-archive-*"))
    (scores_dir / "new.txt").write_text("new")
    instance._save_eval_scores(
        scores_dir=scores_dir,
        extracted_dict={},
        results_df=pd.DataFrame(),
        raw_results_dict={},
    )

    archives = list(tmp_path.glob("scores-archive-*"))
    assert len(archives) == 2
    assert (first_archive / "old.txt").read_text() == "old"
    second_archive = next(archive for archive in archives if archive != first_archive)
    assert (second_archive / "new.txt").read_text() == "new"
