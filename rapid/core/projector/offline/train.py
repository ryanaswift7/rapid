from pathlib import Path
from typing import Any, Dict, Union

import ray.train
import wandb
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)

from rapid.schemas.trainingParams import TrainingParams

from .offlineProjector import OfflineProjectorModule, OffProjModuleConfig
from .projectorDataModule import ProjectorDataModule


def train_projector_per_worker(
    config: Dict[str, Any],
    teacher_cache_dir: Union[str, Path],
    student_cache_dir: Union[str, Path],
    teacher_dim: int,
    student_dim: int,
    alignment_beta: float,
    training_params: TrainingParams,
    max_epochs: int,
    seed: int,
    batch_size: int,
    val_split: float = 0.1,
    test_split: float = 0.1,
):
    """Training function for Ray Tune"""
    if training_params.reproducible_backend:
        ray.train.torch.enable_reproducibility()  # type: ignore

    datamodule = ProjectorDataModule(
        teacher_cache_dir=teacher_cache_dir,
        student_cache_dir=student_cache_dir,
        batch_size=batch_size,
        seed=seed,
        num_workers=training_params.resources.projector.num_data_workers,
        val_split=val_split,
        test_split=test_split,
        pin_memory=True,
        prefetch_factor=4,
        seq_len=training_params.seq_len,
        total_tokens=training_params.offline_projection.total_tokens,
    )
    datamodule.setup(stage="fit")
    cfg = OffProjModuleConfig(
        teacher_dim=teacher_dim,
        student_dim=student_dim,
        alignment_beta=alignment_beta,
        tied_weights=config["tied_weights"],
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        torch_dtype=training_params.torch_dtype,
    )
    model = OfflineProjectorModule(train_config=cfg)
    uses_cuda = training_params.uses_cuda

    trainer = Trainer(
        devices="auto" if config.get("ddp", False) else 1,
        strategy=RayDDPStrategy(),
        plugins=[RayLightningEnvironment()],
        callbacks=[RayTrainReportCallback()],
        max_epochs=max_epochs,
        accelerator=("auto" if uses_cuda and config.get("ddp", False) else ("gpu" if uses_cuda else "cpu")),
        enable_progress_bar=True,
        enable_checkpointing=False,
        deterministic=training_params.reproducible_backend,
        val_check_interval=0.25,
        logger=WandbLogger(
            name=config["wandb_run_name"],
            project=training_params.wandb.project,
            group=training_params.wandb.group,
        ),
    )
    wandb.init(
        project=training_params.wandb.project,
        group=training_params.wandb.group,
        name=config["wandb_run_name"],
        reinit=True,
    )
    trainer = prepare_trainer(trainer)
    trainer.fit(model, datamodule=datamodule)
    wandb.finish()
