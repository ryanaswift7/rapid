import json
import math
import os
import uuid
from typing import Optional, Union

import lightning.pytorch as pl
import ray
import ray.train
import ray.train.torch
import ray.tune
import torch
import torchmetrics
import wandb
from datasets import load_dataset_builder
from lightning.pytorch.loggers import WandbLogger
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

from kdfp.schemas.configBase import ConfigBase
from kdfp.schemas.dataset import DatasetConfig
from kdfp.schemas.directories import ExpInstDirs
from kdfp.schemas.hparams import FixedHparamSet, HparamsConfig, TuneParams, _choice_or_range
from kdfp.schemas.mixins import DtypeSerializerMixin, MCEvalMixin
from kdfp.schemas.trainingParams import TrainingParams
from kdfp.utils.data import load_data
from kdfp.utils.general import log_training_setup


def gen_teacher_weights_filename() -> str:
    return "teacher_ft.bin"


class TeacherModuleConfig(ConfigBase, DtypeSerializerMixin):
    model_name: str
    tokenizer_name: str
    dataset: DatasetConfig
    seq_len: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    max_grad_norm: float
    seed: int
    torch_dtype: Union[str, torch.dtype]
    num_data_workers: int = 0
    n_samples: Optional[int] = None
    total_steps: int
    streaming: bool


class TeacherModule(pl.LightningModule, MCEvalMixin):
    def __init__(self, train_config: TeacherModuleConfig):
        super().__init__()

        # If train_config is a dict (from loading a checkpoint),
        # parse it back into a TeacherTrainConfig object.
        if isinstance(train_config, dict):
            self.train_config = TeacherModuleConfig(**train_config)
        else:
            # Otherwise (on initial creation), use the object directly.
            self.train_config = train_config

        self.save_hyperparameters({"train_config": self.train_config.model_dump()})

        self.model = AutoModelForCausalLM.from_pretrained(
            self.train_config.model_name, dtype=self.train_config.torch_dtype
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.train_config.tokenizer_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
            return_tensors="pt",
        )

        self.val_metrics = torchmetrics.MetricCollection(
            {"ppl": torchmetrics.Perplexity(ignore_index=self.tokenizer.eos_token_id)},
            prefix="val_",
        )
        self.test_metrics = self.val_metrics.clone(prefix="test_")

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def training_step(self, batch, batch_idx):
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)

        val_loss = outputs.loss

        perplexity = torch.exp(val_loss)

        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_ppl", perplexity, prog_bar=True)

        return val_loss

    # def on_validation_epoch_end(self):
    #     self.log_dict(self.val_metrics.compute(), sync_dist=True)
    #     self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        outputs = self.model(**batch)

        test_loss = outputs.loss

        perplexity = torch.exp(test_loss)

        self.log("test_loss", test_loss, prog_bar=True)
        self.log("test_ppl", perplexity, prog_bar=True)

        return test_loss

    # def on_test_epoch_end(self):
    #     self.log_dict(self.test_metrics.compute(), sync_dist=True)
    #     self.test_metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.train_config.learning_rate,
            weight_decay=self.train_config.weight_decay,
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.train_config.total_steps,
            eta_min=1e-9,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def train_dataloader(self):
        dataset = load_data(
            dataset_name=self.train_config.dataset.name,
            subset=self.train_config.dataset.subset,
            tokenizer=self.tokenizer,
            seq_len=self.train_config.seq_len,
            split="train",
            n_samples=self.train_config.n_samples,
            streaming=self.train_config.streaming,
        )

        dataset = dataset.shuffle(seed=self.train_config.seed, buffer_size=1000)  # type: ignore

        return DataLoader(
            dataset,
            batch_size=self.train_config.batch_size,
            shuffle=False,
            num_workers=self.train_config.num_data_workers,
            collate_fn=self.data_collator,
            pin_memory=True,
        )

    def val_dataloader(self):
        dataset = load_data(
            dataset_name=self.train_config.dataset.name,
            subset=self.train_config.dataset.subset,
            tokenizer=self.tokenizer,
            seq_len=self.train_config.seq_len,
            split="validation",
            n_samples=self.train_config.n_samples,
            streaming=self.train_config.streaming,
        )

        return DataLoader(
            dataset,
            batch_size=self.train_config.batch_size,
            shuffle=False,
            num_workers=self.train_config.num_data_workers,
            collate_fn=self.data_collator,
            pin_memory=True,
        )

    def test_dataloader(self):
        dataset = load_data(
            dataset_name=self.train_config.dataset.name,
            subset=self.train_config.dataset.subset,
            tokenizer=self.tokenizer,
            seq_len=self.train_config.seq_len,
            split="test",
            n_samples=self.train_config.n_samples,
            streaming=self.train_config.streaming,
        )

        return DataLoader(
            dataset,
            batch_size=self.train_config.batch_size,
            shuffle=False,
            num_workers=self.train_config.num_data_workers,
            collate_fn=self.data_collator,
            pin_memory=True,
        )

    @classmethod
    def create(
        cls,
        lm_name: str,
        hyperparams: HparamsConfig,
        training_params: TrainingParams,
        dirs: ExpInstDirs,
        seed: int,
    ):
        """Create a teacher for a given task using hyperparameter tuning with Ray Tune"""

        pl.seed_everything(seed, workers=True)
        run_name = "teacher_ft"
        builder = load_dataset_builder(
            training_params.dataset.name, training_params.dataset.subset
        )
        n_train_samples = builder.info.splits["train"].num_examples  # type: ignore
        total_steps = math.ceil(
            (n_train_samples * training_params.num_epochs)
            / training_params.train_batch_size
        )

        def train_teacher_per_worker(config):
            if training_params.reproducible_backend:
                ray.train.torch.enable_reproducibility()
            train_config = TeacherModuleConfig(
                model_name=lm_name,
                tokenizer_name=lm_name,
                dataset=training_params.dataset,
                seq_len=training_params.seq_len,
                batch_size=training_params.train_batch_size,
                learning_rate=config["learning_rate"],
                weight_decay=config["weight_decay"],
                max_grad_norm=config["max_grad_norm"],
                n_samples=training_params.total_tokens,
                num_data_workers=training_params.resources.teacher.num_data_workers,
                seed=seed,
                torch_dtype=training_params.torch_dtype,
                total_steps=total_steps,
                streaming=training_params.resources.stream_task_data,
            )

            module = TeacherModule(train_config=train_config)
            uses_cuda = training_params.uses_cuda

            if config.get("ddp", False):
                trainer = pl.Trainer(
                    devices="auto",
                    accelerator="auto" if uses_cuda else "cpu",
                    strategy=RayDDPStrategy(),
                    plugins=[RayLightningEnvironment()],
                    callbacks=[RayTrainReportCallback()],
                    max_epochs=training_params.num_epochs,
                    gradient_clip_algorithm="norm",
                    gradient_clip_val=config["max_grad_norm"],
                    enable_progress_bar=False,
                    enable_checkpointing=False,
                    deterministic=training_params.reproducible_backend,
                    check_val_every_n_epoch=1,
                    logger=WandbLogger(
                        name=config["wandb_run_name"],
                        project=training_params.wandb.project,
                        group=training_params.wandb.group,
                    ),
                )
            else:
                trainer = pl.Trainer(
                    devices=1,
                    accelerator="gpu" if uses_cuda else "cpu",
                    strategy=RayDDPStrategy(),
                    plugins=[RayLightningEnvironment()],
                    callbacks=[RayTrainReportCallback()],
                    max_epochs=training_params.num_epochs,
                    gradient_clip_algorithm="norm",
                    gradient_clip_val=config["max_grad_norm"],
                    enable_progress_bar=False,
                    enable_checkpointing=False,
                    deterministic=training_params.reproducible_backend,
                    check_val_every_n_epoch=1,
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
            module.train()
            trainer.fit(module)
            wandb.finish()

        def train_fixed_params(fixed_params: FixedHparamSet):
            visible_gpus = training_params.resources.visible_gpus if training_params.uses_cuda else 0
            role_resources = training_params.resources.teacher
            num_ray_workers = role_resources.resolve_num_ray_workers(
                fixed_mode=True,
                visible_gpus=visible_gpus,
            )
            num_cpus_per_trial = role_resources.resolve_num_cpus_per_trial(visible_gpus)
            fixed_config = {
                "learning_rate": fixed_params.lr,
                "weight_decay": fixed_params.weight_decay,
                "max_grad_norm": fixed_params.max_grad_norm,
                "ddp": True,
                "wandb_run_name": f"{run_name}_fixed_{uuid.uuid4().hex[:6]}",
            }
            scaling_config = ray.train.ScalingConfig(
                num_workers=num_ray_workers,
                use_gpu=training_params.uses_cuda,
                resources_per_worker={
                    "CPU": num_cpus_per_trial,
                    **({"GPU": 1.0} if training_params.uses_cuda else {}),
                },
                placement_strategy="PACK",
            )
            log_training_setup(
                component="teacher",
                mode="fixed",
                visible_gpus=visible_gpus,
                num_workers=num_ray_workers,
                num_gpus_per_trial=1.0 if training_params.uses_cuda else 0.0,
                num_cpus_per_trial=num_cpus_per_trial,
            )
            train_run_config = ray.train.RunConfig(
                name=fixed_config["wandb_run_name"],
                checkpoint_config=ray.train.CheckpointConfig(
                    num_to_keep=1,
                    checkpoint_frequency=0,
                    checkpoint_score_attribute="val_ppl",
                    checkpoint_score_order="min",
                ),
                failure_config=ray.train.FailureConfig(
                    max_failures=training_params.resources.max_failures
                ),
                storage_path=str(dirs.ray),
            )

            trainer = ray.train.torch.TorchTrainer(
                train_teacher_per_worker,
                train_loop_config=fixed_config,
                scaling_config=scaling_config,
                run_config=train_run_config,
            )

            return trainer.fit()

        def tune_asha(num_trials: int, tune_params: TuneParams):
            visible_gpus = training_params.resources.visible_gpus if training_params.uses_cuda else 0
            role_resources = training_params.resources.teacher
            num_ray_workers = role_resources.resolve_num_ray_workers(
                fixed_mode=False,
                visible_gpus=visible_gpus,
            )
            num_cpus_per_trial = role_resources.resolve_num_cpus_per_trial(visible_gpus)
            # cannot quantize loguniform with optuna
            tune_config = {
                "learning_rate": _choice_or_range(tune_params.lr, kind="qloguniform", quantization=1e-6),
                "weight_decay": _choice_or_range(tune_params.weight_decay, kind="qloguniform", quantization=1e-4),
                "max_grad_norm": _choice_or_range(tune_params.max_grad_norm, kind="quniform", quantization=1e-1),
                "ddp": False,
                "wandb_run_name": f"{run_name}_tune_{uuid.uuid4().hex[:6]}",
            }
            scaling_config = ray.train.ScalingConfig(
                num_workers=num_ray_workers,
                use_gpu=training_params.uses_cuda,
                resources_per_worker={
                    "CPU": num_cpus_per_trial,
                    **({"GPU": 1.0} if training_params.uses_cuda else {}),
                },
                placement_strategy="PACK",
            )
            log_training_setup(
                component="teacher",
                mode="tune",
                visible_gpus=visible_gpus,
                num_workers=num_ray_workers,
                num_gpus_per_trial=1.0 if training_params.uses_cuda else 0.0,
                num_cpus_per_trial=num_cpus_per_trial,
            )
            tune_run_config = ray.tune.RunConfig(
                name=tune_config["wandb_run_name"],
                checkpoint_config=ray.tune.CheckpointConfig(
                    num_to_keep=1,
                    checkpoint_frequency=0,
                    checkpoint_score_attribute="val_ppl",
                    checkpoint_score_order="min",
                ),
                failure_config=ray.tune.FailureConfig(
                    max_failures=training_params.resources.max_failures
                ),
                storage_path=str(dirs.ray),
                callbacks=[
                    WandbLoggerCallback(
                        project=training_params.wandb.project,
                        group=training_params.wandb.group,
                        run_name=tune_config["wandb_run_name"],
                    )
                ],
            )
            ray_ptl_trainer = ray.train.torch.TorchTrainer(
                train_teacher_per_worker,
                scaling_config=scaling_config,
            )

            optuna_search = OptunaSearch(
                metric="val_ppl",
                mode="min",
                seed=seed,
            )
            search_alg = ConcurrencyLimiter(
                optuna_search,
                max_concurrent=math.floor(
                    max(1, visible_gpus)
                ),
            )

            asha_scheduler = ASHAScheduler(
                time_attr="training_iteration",
                max_t=total_steps,
                grace_period=1,
                reduction_factor=3,
            )

            tuner = ray.tune.Tuner(
                trainable=ray_ptl_trainer,  # type: ignore
                param_space={"train_loop_config": tune_config},
                run_config=tune_run_config,
                tune_config=ray.tune.TuneConfig(
                    metric="val_ppl",
                    mode="min",
                    num_samples=num_trials,
                    scheduler=asha_scheduler,
                    search_alg=search_alg,
                ),
            )
            return tuner.fit()

        fixed_params = hyperparams.fixed.teacher if hyperparams.fixed else None

        if fixed_params:
            best_result = train_fixed_params(fixed_params=fixed_params)
        else:
            assert hyperparams.tune, "No hyperparameters provided for tuning."
            best_result = tune_asha(
                num_trials=training_params.n_trials, tune_params=hyperparams.tune
            ).get_best_result()

        assert best_result.checkpoint is not None, (
            "No checkpoint found in best Teacher result."
        )
        with best_result.checkpoint.as_directory() as path:
            return TeacherModule.load_from_checkpoint(f"{path}/checkpoint.ckpt")

    @property
    def model_file(self) -> str:
        return gen_teacher_weights_filename()

    def save_pretrained(self, path: str) -> None:
        """Save the model and tokenizer"""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        save_path = f"{path}/{self.model_file}"
        torch.save(self.model.state_dict(), save_path)

        with open(f"{path}/pl_config.json", "w") as f:
            json.dump(self.train_config.model_dump(), f, indent=4)

        print(f"Finetuned Teacher saved to {path}")

    @classmethod
    def load_pretrained(cls, path: str) -> "TeacherModule":
        with open(f"{path}/pl_config.json", "r") as f:
            hparams = json.load(f)
        cfg = TeacherModuleConfig(**hparams)
        teacher_module = cls(train_config=cfg)
        teacher_module.model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=teacher_module.train_config.torch_dtype
        )
        teacher_module.train_config.model_name = path
        teacher_module.tokenizer = AutoTokenizer.from_pretrained(path)

        print(f"Finetuned Teacher loaded from {path}")
        return teacher_module
