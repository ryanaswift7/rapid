import os
from typing import Literal, Optional

import torch
from pydantic import computed_field

from kdfp.schemas.configBase import ConfigBase
from kdfp.schemas.dataset import DatasetConfig
from kdfp.schemas.mixins import DtypeSerializerMixin
from kdfp.schemas.projections import OffProjTrainArgs


class Resources(ConfigBase):
    num_data_workers: int = 0
    num_ray_workers: Optional[int] = None
    num_cpus_per_trial: Optional[int] = None
    num_gpus_per_trial: Optional[float] = None

    def resolve_num_ray_workers(self, fixed_mode: bool, visible_gpus: int) -> int:
        return max(1, visible_gpus) if fixed_mode else 1

    def resolve_num_cpus_per_trial(self, visible_gpus: int) -> int:
        return self.num_cpus_per_trial if self.num_cpus_per_trial is not None else 16

    def resolve_num_gpus_per_trial(self, fixed_mode: bool) -> float:
        return 1.0


class ExperimentResources(ConfigBase):
    max_failures: int
    num_available_gpus: Optional[int] = None
    stream_task_data: bool
    projector: Resources
    teacher: Resources
    student: Resources

    @property
    def visible_gpus(self) -> int:
        if not torch.cuda.is_available():
            return 0

        gpu_count = torch.cuda.device_count()
        if gpu_count <= 0:
            raise RuntimeError("No visible CUDA GPUs detected.")

        return gpu_count


class TrainingControls(ConfigBase):
    finetune_teacher: bool
    get_baselines: list[str]
    run_or_test: Literal["run", "test"]


class WandbParams(ConfigBase):
    project: str
    group: str


class TrainingParams(ConfigBase, DtypeSerializerMixin):
    dataset: DatasetConfig
    seq_len: int
    train_batch_size: int
    test_batch_size: int
    total_tokens: Optional[int] = None
    num_epochs: int
    n_trials: int
    reproducible_backend: bool
    device: str
    torch_dtype: torch.dtype
    controls: TrainingControls
    resources: ExperimentResources
    offline_projection: OffProjTrainArgs
    wandb: WandbParams

    @computed_field
    @property
    def num_samples(self) -> Optional[int]:
        if self.total_tokens and self.seq_len:
            return self.total_tokens // self.seq_len
        else:
            return None

    @property
    def uses_cuda(self) -> bool:
        return self.device.startswith("cuda")
