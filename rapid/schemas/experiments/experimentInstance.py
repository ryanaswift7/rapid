import gc
import json
import shutil
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import lightning.pytorch as pl
import pandas as pd
import torch
from transformers import AutoConfig, AutoTokenizer

from rapid.core.projector import OfflineProjectorModule
from rapid.core.projector.offline.offlineProjector import gen_off_proj_weights_filename
from rapid.core.student.student import StudentModule, gen_student_weights_filename
from rapid.core.teacher import TeacherModule, gen_teacher_weights_filename
from rapid.schemas.configBase import ConfigBase
from rapid.schemas.directories import ExpInstDirs
from rapid.schemas.experiments.experimentConfig import ExperimentConfig
from rapid.schemas.hparams import HparamsConfig
from rapid.schemas.tasks import LMEvalTasks
from rapid.schemas.trainingParams import TrainingParams
from rapid.utils.activation_caching import (
    cache_student_and_teacher_acts,
)
from rapid.utils.enums import ProjectionTraining
from rapid.utils.evals import evaluate_model, save_eval_scores


class ExperimentInstance(ConfigBase):
    cfg: ExperimentConfig
    dirs: ExpInstDirs
    training_params: TrainingParams
    hparams: HparamsConfig
    tasks: LMEvalTasks

    # only offline projectors are initialized here bc don't want to pass in models to
    # lightning modules, so I load them in the student module
    def _init_offline_projector(
        self,
        # cache_dir: str,
        # save_dir: str,
        local_teacher_dir: Optional[str] = None,
        local_student_dir: Optional[str] = None,
    ) -> None:
        """Initialize the offline projector for an experiment. Defaults to pretrained teacher
        and student, but you may use custom local versions instead by passing in the directories.
        """
        # if we already trained one, load it instead of creating another
        off_proj_weights_path = Path(
            self.dirs.off_proj_models  # type: ignore
        ) / gen_off_proj_weights_filename(
            alignment_beta=self.cfg.projection.offline_alignment_beta  # type: ignore
        )
        if off_proj_weights_path.exists():
            print(
                f"Pretrained offline projector for with alignment beta '{self.cfg.projection.offline_alignment_beta}' located at: {str(off_proj_weights_path)}"  # type: ignore
            )

            # p = OfflineProjectorModule.load_pretrained(self.dirs.off_proj_model)
            # return p.extract_single_direction(direction=self.projection.direction)
        else:
            # cache dir should always be specified for offline projector
            if cache_dir := self.dirs.cache:
                (teacher_cache_dir, teacher_dim, student_cache_dir, student_dim) = (
                    cache_student_and_teacher_acts(
                        cfg=self.cfg,
                        training_params=self.training_params,
                        cache_dir=str(cache_dir),
                        cache_outputs=False,
                        local_teacher_dir=local_teacher_dir,
                        local_student_dir=local_student_dir,
                    )
                )

            # create projector
            projector = OfflineProjectorModule.create(
                teacher_cache_dir=teacher_cache_dir,
                student_cache_dir=student_cache_dir,
                teacher_dim=teacher_dim,
                student_dim=student_dim,
                projection=self.cfg.projection,  # type: ignore
                hyperparams=self.hparams,
                training_params=self.training_params,
                dirs=self.dirs,
                val_split=0.1,
                test_split=0.1,
                seed=self.cfg.seed,
            )
            projector.save_pretrained(path=self.dirs.off_proj_models)  # type: ignore
            # load_test = OfflineProjector.load_pretrained(save_path=self.dirs.off_proj_models)
            # return projector.extract_single_direction(direction=self.projection.direction)

    def init_projector(self) -> None:
        """Initialize projector for an experiment. kwargs are for init_offline_projector."""
        if proj := self.cfg.projection:
            if proj.training_method == ProjectionTraining.ONLINE:
                print(
                    f"{proj.training_method.value.upper()} projector will be initalized during the experiment."
                )

            elif proj.training_method == ProjectionTraining.OFFLINE:
                print(
                    f"{proj.training_method.value.upper()} projector initializtion starting..."
                )
                self._init_offline_projector()
                print(
                    f"{proj.training_method.value.upper()} projector initializtion complete!"
                )

            elif proj.training_method == ProjectionTraining.RANDOMIZED:
                print(
                    f"{proj.training_method.value.upper()} projector will be initalized during the experiment."
                )

            else:
                raise ValueError(
                    f"Invalid projection training specified: {proj.training_method}."
                )

        return

    def init_ft_teacher(self) -> None:
        teacher_weights_file = (
            Path(self.dirs.teacher_model) / gen_teacher_weights_filename()
        )
        if teacher_weights_file.exists():
            print(f"Pretrained teacher located at: {str(teacher_weights_file)}")
        else:
            t = TeacherModule.create(
                lm_name=self.cfg.distill_pair.teacher.name,
                hyperparams=self.hparams,
                training_params=self.training_params,
                dirs=self.dirs,
                seed=self.cfg.seed,
            )
            t.save_pretrained(path=f"{self.dirs.teacher_model}/{t.model_file}")

    def init(self) -> None:
        """Initialize finetuned teachers and projectors (if necessary)"""
        if self.cfg.projection:
            self.init_projector()
        if self.training_params.controls.finetune_teacher:
            self.init_ft_teacher()

    def cleanup_log_dirs(self) -> None:
        shutil.rmtree(self.dirs.ray, ignore_errors=True)
        shutil.rmtree(self.dirs.lightning, ignore_errors=True)

    def _student_model_exists(self, model_save_dir: str) -> bool:
        path = Path(model_save_dir)

        required_paths = [
            path / "pl_config.json",
            path / gen_student_weights_filename(),
            path / "config.json",
        ]
        if not all(p.exists() for p in required_paths):
            return False

        tokenizer_markers = [
            path / "tokenizer_config.json",
            path / "tokenizer.json",
            path / "special_tokens_map.json",
        ]
        if not any(p.exists() for p in tokenizer_markers):
            return False

        try:
            AutoConfig.from_pretrained(path)
            AutoTokenizer.from_pretrained(path)
        except Exception:
            return False

        return True

    def _eval_scores_paths(self, scores_dir: Path) -> dict[str, Path]:
        return {
            "raw": scores_dir / "conventional_raw_eval_scores.json",
            "extracted": scores_dir / "conventional_extracted_eval_scores.json",
            "csv": scores_dir / "results_df.csv",
            "parquet": scores_dir / "results_df.parquet",
        }

    def _load_cached_eval_scores(
        self, scores_dir: Path
    ) -> tuple[dict, pd.DataFrame, dict] | None:
        paths = self._eval_scores_paths(scores_dir)
        if not paths["raw"].exists() or not paths["extracted"].exists():
            return None

        with open(paths["raw"], "r") as f:
            raw_results_dict = json.load(f)

        with open(paths["extracted"], "r") as f:
            extracted_dict = json.load(f)

        if paths["parquet"].exists():
            results_df = pd.read_parquet(paths["parquet"])
        elif paths["csv"].exists():
            results_df = pd.read_csv(paths["csv"])
        else:
            results_df = pd.DataFrame()

        return extracted_dict, results_df, raw_results_dict

    def _save_eval_scores(
        self,
        scores_dir: Path,
        extracted_dict: dict,
        results_df: pd.DataFrame,
        raw_results_dict: dict,
    ) -> None:
        if self.tasks.force_rerun and scores_dir.exists():
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            archive_dir = scores_dir.with_name(f"{scores_dir.name}-archive-{timestamp}")
            suffix = 1
            while archive_dir.exists():
                archive_dir = scores_dir.with_name(
                    f"{scores_dir.name}-archive-{timestamp}-{suffix}"
                )
                suffix += 1
            scores_dir.rename(archive_dir)
            print(f"Archived previous eval scores to {archive_dir}")

        save_eval_scores(
            scores_dir=scores_dir,
            extracted_dict=extracted_dict,
            results_df=results_df,
            raw_results_dict=raw_results_dict,
        )

    def test_trainer_factory(self) -> pl.Trainer:
        return pl.Trainer(
            accelerator="gpu" if self.training_params.uses_cuda else "cpu",
            strategy="auto",
            devices=1,
            deterministic=self.training_params.reproducible_backend,
            default_root_dir=self.dirs.lightning,
        )

    def run(self) -> List[dict]:
        # DO NOT OVERWRITE EXISTING MODELS
        # student_already_exists = False
        model_save_dir = str(self.dirs.student_model)
        # if os.path.isdir(model_save_dir) and os.listdir(model_save_dir):
        #     warnings.warn(
        #         "Student model directory exists and is not empty!\n"
        #         + f"Skipping experiment {self.cfg.name.upper()} to avoid overwriting models..."
        #     )
        #     student_already_exists = True

        student_already_exists = self._student_model_exists(model_save_dir)
        if student_already_exists:
            warnings.warn(
                "Student already exists!\n"
                + f"Skipping experiment {self.cfg.name.upper()} to avoid overwriting models..."
            )

        if not student_already_exists:
            print(f"Training student {self.cfg.name}'...")
            student_module = StudentModule.create(
                cfg=self.cfg,
                dirs=self.dirs,
                hyperparams=self.hparams,
                training_params=self.training_params,
                eval_tasks=self.tasks,
            )
            student_module.save_pretrained(model_save_dir)
            print(f"Student {self.cfg.name} training complete!")

        return self.test_student()

    def test_student(self) -> List[dict]:
        """Evaluate a student model with task-specific few-shot settings."""
        print(f"Running tests on student {self.cfg.name}...")

        student_model_path = str(self.dirs.student_model)
        scores_dir = Path(f"{student_model_path}/scores")
        try:
            cached = (
                None
                if self.tasks.force_rerun
                else self._load_cached_eval_scores(scores_dir)
            )

            if cached is not None:
                extracted_dict, results_df, raw_results_dict = cached
                print(f"Loaded cached eval scores from {scores_dir}")
            else:
                print(f"\nBenchmarking tasks with configured per-task few-shot settings...")
                print(f"Tasks: {', '.join(self.tasks.names)}")

                extracted_dict, results_df, raw_results_dict = evaluate_model(
                    model_path=student_model_path,
                    tasks=self.tasks.names,
                    # The parent directory is the canonical name shared by all
                    # seed-specific model directories.
                    experiment_name=Path(student_model_path).parent.name,
                    batch_size=self.training_params.test_batch_size,
                    device=self.training_params.device,
                    limit=self.tasks.sample_limit,
                    seed=self.cfg.seed,
                )

                self._save_eval_scores(
                    scores_dir=scores_dir,
                    extracted_dict=extracted_dict,
                    results_df=results_df,
                    raw_results_dict=raw_results_dict,
                )

            print("\nFinished benchmarking all tasks!")
            print(f"Test scores can be found at: {scores_dir}")
            print(f"Testing of student {self.cfg.name} complete!")
            print("=========================================")

            return results_df.to_dict(orient="records") if not results_df.empty else []
        finally:
            self.cleanup_log_dirs() # Assumes this class method exists
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def test_teacher(self, seeds: List[int]) -> List[dict]:
        """Evaluate a teacher model with task-specific few-shot settings."""
        teacher_name = self.cfg.distill_pair.teacher.name

        print(f"Running tests on teacher model {teacher_name}...")

        all_results = []
        try:
            for seed in seeds:
                scores_dir = Path(f"{self.dirs.teacher_model}/seed{seed}/scores")
                cached = (
                    None
                    if self.tasks.force_rerun
                    else self._load_cached_eval_scores(scores_dir)
                )

                if cached is not None:
                    extracted_dict, results_df, raw_results_dict = cached
                    print(f"Loaded cached eval scores from {scores_dir}")
                else:
                    print(f"\nBenchmarking tasks with configured per-task few-shot settings...")
                    print(f"Tasks: {', '.join(self.tasks.names)}")

                    extracted_dict, results_df, raw_results_dict = evaluate_model(
                        model_path=teacher_name,
                        tasks=self.tasks.names,
                        experiment_name=f"teacher-{teacher_name}",
                        batch_size=self.training_params.test_batch_size,
                        device=self.training_params.device,
                        limit=self.tasks.sample_limit,
                        seed=seed,
                    )

                    self._save_eval_scores(
                        scores_dir=scores_dir,
                        extracted_dict=extracted_dict,
                        results_df=results_df,
                        raw_results_dict=raw_results_dict,
                    )

                print(f"\nFinished benchmarking all tasks for seed: {seed}")

                if not results_df.empty:
                    all_results.extend(results_df.to_dict(orient="records"))

            print(f"Testing of teacher {teacher_name} complete!")
            print("=========================================")

            return all_results
        finally:
            self.cleanup_log_dirs()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
