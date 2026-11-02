from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pydantic import model_validator

from rapid.schemas.configBase import ConfigBase
from rapid.schemas.experiments.experimentConfig import ExperimentConfig
from rapid.schemas.mixins import PathSerializerMixin
from rapid.utils.enums import ProjectionTraining


class ExpInstDirs(ConfigBase, PathSerializerMixin):
    teacher_model: Path
    student_model: Path
    ray: Path
    lightning: Path
    suite_scores: Path
    cache: Optional[Path] = None
    off_proj_models: Optional[Path] = None

    # @computed_field
    # @property
    # def teacher_scores(self) -> Path:
    #     return self.teacher_model / "scores"

    # @computed_field
    # @property
    # def student_scores(self) -> Path:
    #     return self.student_model / "scores"

    @model_validator(mode="after")
    def resolve_paths(self) -> "ExpInstDirs":
        """
        Converts all path fields to absolute, resolved paths.
        Using .resolve() handles symlinks and removes '..' or '.' segments.
        """
        # We can iterate over the fields to avoid repeating 'os.path.abspath' 8 times
        for field_name, value in self.__dict__.items():
            if isinstance(value, Path):
                # .resolve() is the pathlib equivalent of abspath + realpath
                setattr(self, field_name, value.resolve())
            elif value is not None and field_name in self.model_fields:
                # This catches any edge cases where a field might be a string
                setattr(self, field_name, Path(value).resolve())

        return self


@dataclass
class BaseDirs:
    base_cache_dir: Path
    base_model_dir: Path
    ray_dir: Path
    lightning_dir: Path
    suite_root_dir: Path

    def create_instance_dirs(self, exp_cfg: ExperimentConfig) -> "ExpInstDirs":
        """
        Maps base directories to experiment-specific paths using dynamic pair lookup.
        """

        if proj := exp_cfg.projection:
            is_offline = (
                proj is not None and proj.training_method == ProjectionTraining.OFFLINE
            )
            cache_path = self.base_cache_dir if is_offline else None

            off_proj_path = None
            if proj.offline_alignment_beta is not None:
                off_proj_path = (
                    self.base_model_dir
                    / "offline_projectors"
                    / f"off_proj_{proj.offline_alignment_beta}"
                )
        else:
            cache_path = None
            off_proj_path = None

        self.base_model_dir.mkdir(parents=True, exist_ok=True)
        return ExpInstDirs(
            teacher_model=self.base_model_dir
            / "teachers"
            / exp_cfg.distill_pair.teacher.name,
            # / f"seed{exp_cfg.seed}",
            student_model=self.base_model_dir / "students" / exp_cfg.name / f"seed{exp_cfg.seed}",
            ray=self.ray_dir,
            lightning=self.lightning_dir,
            cache=cache_path,
            off_proj_models=off_proj_path / f"seed{exp_cfg.seed}" if off_proj_path is not None else off_proj_path,
            suite_scores=self.suite_root_dir / "scores",
        )


class DirectoryParser(ConfigBase):
    root: str
    cache: str
    models: str
    ray: str
    lightning: str

    def resolve(self, suite_title: str) -> BaseDirs:
        """Operates on fields to instantiate a BaseDirs object with absolute Paths."""
        # Convert root to a Path once; resolve() makes it absolute
        root_path = Path(self.root).resolve()
        suite_root = root_path / suite_title
        suite_root.mkdir(parents=True, exist_ok=True)

        return BaseDirs(
            base_cache_dir=root_path / self.cache,
            base_model_dir=suite_root / self.models,
            ray_dir=root_path / self.ray,
            lightning_dir=root_path / self.lightning,
            suite_root_dir=suite_root,
        )

    def get_suite_scores_dir(self, suite_title: str) -> Path:
        """Returns the resolved path for the suite scores directory."""
        root_path = Path(self.root).resolve()
        return root_path / suite_title / "scores"
