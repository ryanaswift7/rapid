from typing import List, Literal, Optional

from rapid.schemas.configBase import ConfigBase
from rapid.schemas.dataset import DatasetConfig
from rapid.utils.enums import ProjectionDirection, ProjectionTraining


class OffProjTrainArgs(ConfigBase):
    dataset: DatasetConfig
    total_tokens: Optional[int] = None
    cache_batch_size: int
    train_batch_size: int
    num_epochs: int
    n_trials: int
    force_cache_all: bool


class ProjectionInstance(ConfigBase):
    direction: ProjectionDirection
    training_method: ProjectionTraining
    offline_alignment_beta: Optional[float] = None
    init_mode: Literal["per-component", "shared"] = "shared"


class ProjectionParser(ConfigBase):
    directions: List[ProjectionDirection]
    training_methods: List[ProjectionTraining]
    offline_alignment_betas: List[float]
    init_mode: Literal["per-component", "shared"] = "shared"
