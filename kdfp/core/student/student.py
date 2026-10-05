import json
import math
import os
import uuid
import tempfile
import shutil
from pathlib import Path
from typing import Optional

import lightning.pytorch as pl
import ray
import ray.train
import ray.train.torch
import ray.tune
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from datasets import load_dataset_builder
from lightning.pytorch.loggers import WandbLogger
from ray.train import Checkpoint
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
from transformer_lens import HookedTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

from kdfp.core.projector.offline.offlineProjector import OfflineProjectorModule
from kdfp.core.projector.online.onlineProjector import OnlineProjector
from kdfp.core.projector.randomized.RandomizedProjector import RandomizedProjector
from kdfp.core.student.training_utils import forward_kl_divergence
from kdfp.core.teacher import gen_teacher_weights_filename
from kdfp.schemas.configBase import ConfigBase
from kdfp.schemas.dataset import DatasetConfig
from kdfp.schemas.directories import ExpInstDirs
from kdfp.schemas.experiments.experimentConfig import ExperimentConfig
from kdfp.schemas.hparams import (
    FixedHparamSet,
    HparamsConfig,
    TuneParams,
    _choice_or_range,
    get_fixed_student_params,
)
from kdfp.schemas.tasks import LMEvalTasks
from kdfp.schemas.mixins import DtypeSerializerMixin, MCEvalMixin
from kdfp.schemas.models import DistillationPair
from kdfp.schemas.objective_weights import ObjectiveWeights
from kdfp.schemas.projections import ProjectionInstance
from kdfp.schemas.trainingParams import TrainingParams
from kdfp.utils.data import load_data
from kdfp.utils.enums import (
    ObjectiveWeightStr,
    ProjectionDirection,
    ProjectionTraining,
    TrainingType,
)
from kdfp.utils.evals import evaluate_model, mean_task_score, save_eval_scores
from kdfp.utils.general import log_training_setup
from kdfp.utils.shallow_model_loading import load_shallow
from kdfp.utils.weight_conversions import convert_hf_to_hooked_weights, convert_hooked_to_hf_weights
from kdfp.schemas.dynamic_objective_weights import (
    DynamicObjectiveLossPL,
    RawObjectiveLosses,
    StaticObjectiveLossPL,
)


def gen_student_weights_filename() -> str:
    return "student.bin"


class StudentModuleConfig(ConfigBase, DtypeSerializerMixin):
    exp_name: str
    distill_pair: DistillationPair
    dirs: ExpInstDirs
    projection: Optional[ProjectionInstance] = None
    dataset: DatasetConfig
    seq_len: int
    batch_size: int
    temperature: float
    learning_rate: float
    weight_decay: float
    max_grad_norm: float
    seed: int
    torch_dtype: torch.dtype
    num_data_workers: int = 0
    total_tokens: Optional[int]
    num_samples: Optional[int]
    objective_weights: Optional[ObjectiveWeights]
    objective_variant: Optional[ObjectiveWeightStr] = None
    total_steps: int
    finetune_teacher: bool
    training_type: TrainingType
    streaming: bool
    device: str = "cpu"


class StudentModule(pl.LightningModule, MCEvalMixin):
    def __init__(self, train_config):
        super().__init__()

        # If train_config is a dict (from loading a checkpoint),
        # parse it back into a TeacherTrainConfig object.
        if isinstance(train_config, dict):
            self.train_config = StudentModuleConfig(**train_config)
        else:
            # Otherwise (on initial creation), use the object directly.
            self.train_config = train_config
        # self.train_config = train_config

        # Now self.train_config is guaranteed to be a TeacherTrainConfig object,
        # and you can safely call .model_dump()
        self.save_hyperparameters({"train_config": self.train_config.model_dump()})

        # self.tokenizer = AutoTokenizer.from_pretrained(self.train_config.tokenizer_name)
        self.teacher = self.load_teacher()
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        self.student = self.load_student()
        self.tokenizer = self.student.tokenizer
        self.projector = self.load_projector()

        # Add padding token if it doesn't exist
        assert self.tokenizer is not None, "Tokenizer could not be loaded."
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Data collator for language modeling
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
            return_tensors="pt",
        )
        self.mse = nn.MSELoss()
        active_objectives = self._effective_active_objectives()
        if self._uses_dynamic_objective():
            self.loss_fn = DynamicObjectiveLossPL(active_objectives=active_objectives)
        else:
            self.loss_fn = StaticObjectiveLossPL(active_objectives=active_objectives)

        # self.val_metrics = torchmetrics.MetricCollection(
        #     {"ppl": torchmetrics.Perplexity(ignore_index=self.tokenizer.eos_token_id)},
        #     prefix="val_",
        # )
        # self.test_metrics = self.val_metrics.clone(prefix="test_")

    def train(self, mode: bool = True):
        """Train the student while keeping the teacher frozen and in eval mode."""
        super().train(mode)
        if hasattr(self, "teacher"):
            self.teacher.eval()
            self.teacher.requires_grad_(False)
        return self

    def load_teacher(self) -> HookedTransformer:
        if self.train_config.finetune_teacher:
            teacher_hf = AutoModelForCausalLM.from_pretrained(
                f"{self.train_config.dirs.teacher_model}/{gen_teacher_weights_filename()}",
            )
            # load base teacher
            teacher_tl = HookedTransformer.from_pretrained_no_processing(
                self.train_config.distill_pair.teacher.name,
                dtype=self.train_config.torch_dtype,
                device=self.train_config.device,
            )

            print("Loading finetuned weights for teacher model...")
            # load finetuned weights, convert to transformer lens format, load into teacher
            tl_teacher_weights = convert_hf_to_hooked_weights(
                teacher_hf, cfg=teacher_tl.cfg
            )
            teacher_tl.load_state_dict(tl_teacher_weights, strict=False)
        else:
            teacher_tl = HookedTransformer.from_pretrained_no_processing(
                self.train_config.distill_pair.teacher.name,
                dtype=self.train_config.torch_dtype,
                device=self.train_config.device,
            )
        return teacher_tl.eval()

    def load_student(self) -> HookedTransformer:
        if self.train_config.distill_pair.student.layer_reduction_factor:
            student, tokenizer, keep_indices = load_shallow(
                model_name=self.train_config.distill_pair.student.name,
                mode="tl",
                layer_reduction_factor=self.train_config.distill_pair.student.layer_reduction_factor,
                selection_method="linspace",
                torch_dtype=self.train_config.torch_dtype,
                device=self.train_config.device,
            )
        else:
            student = HookedTransformer.from_pretrained_no_processing(
                self.train_config.distill_pair.student.name,
                dtype=self.train_config.torch_dtype,
                device=self.train_config.device,
            )
            tokenizer = student.tokenizer

        tokenizer_name = self.train_config.distill_pair.student.tokenizer_name
        if tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        student.tokenizer = tokenizer
        return student

    def _create_projector(self) -> torch.nn.Module:
        projection = self.train_config.projection
        assert projection is not None

        if projection.training_method == ProjectionTraining.OFFLINE:
            p = OfflineProjectorModule.load_pretrained(
                f"{self.train_config.dirs.off_proj_models}"
            )
            return p.extract_single_direction(projection.direction).eval()
        if projection.training_method == ProjectionTraining.ONLINE:
            return OnlineProjector(
                teacher_dim=self.train_config.distill_pair.teacher.d_model,
                student_dim=self.train_config.distill_pair.student.d_model,
                direction=projection.direction,
                torch_dtype=self.train_config.torch_dtype,
            ).train()
        if projection.training_method == ProjectionTraining.RANDOMIZED:
            return RandomizedProjector(
                teacher_dim=self.train_config.distill_pair.teacher.d_model,
                student_dim=self.train_config.distill_pair.student.d_model,
                direction=projection.direction,
                torch_dtype=self.train_config.torch_dtype,
            ).eval()
        raise ValueError(
            f"Unrecognized projection training method: {projection.training_method.value.upper()}"
        )

    @staticmethod
    def _projector_key(teacher_hook: str, student_hook: str) -> str:
        return f"{teacher_hook}__{student_hook}".replace(".", "_")

    def _active_activation_components(self) -> frozenset[str]:
        return frozenset(
            self._effective_active_objectives()
            & {"embed", "attn_out", "mlp", "resid"}
        )

    def _effective_active_objectives(self) -> frozenset[str]:
        """Return objectives enabled for this student's training mode."""
        projection = self.train_config.projection
        if projection is None:
            return frozenset(("divergence", "true_labels"))

        weights = self.train_config.objective_weights
        if weights is None:
            return frozenset(("divergence", "true_labels"))
        return frozenset(
            weights.active_objectives
        )

    def _uses_dynamic_objective(self) -> bool:
        projection = self.train_config.projection
        return (
            projection is not None
            and projection.training_method == ProjectionTraining.RANDOMIZED
        )

    def _active_hook_pairs(self) -> list[tuple[str, str]]:
        teacher_hooks = self.train_config.distill_pair.teacher.hook_list
        student_hooks = self.train_config.distill_pair.student.hook_list
        if len(teacher_hooks) != len(student_hooks):
            raise ValueError("Teacher and student hook lists must have the same length.")

        active_components = self._active_activation_components()
        return [
            (teacher_hook, student_hook)
            for teacher_hook, student_hook in zip(teacher_hooks, student_hooks)
            if any(component in student_hook for component in active_components)
        ]

    def load_projector(self) -> torch.nn.Module:
        projection = self.train_config.projection
        if projection is None:
            return nn.Identity()

        if not self._active_activation_components():
            return nn.Identity()

        if projection.init_mode == "shared":
            return self._create_projector()

        if projection.training_method == ProjectionTraining.OFFLINE:
            raise NotImplementedError(
                "Per-component initialization is not implemented for offline projectors."
            )

        projectors = nn.ModuleDict()
        for teacher_hook, student_hook in self._active_hook_pairs():
            projectors[self._projector_key(teacher_hook, student_hook)] = (
                self._create_projector()
            )
        return projectors

    def _projector_for_hooks(self, teacher_hook: str, student_hook: str) -> nn.Module:
        if not isinstance(self.projector, nn.ModuleDict):
            return self.projector
        return self.projector[self._projector_key(teacher_hook, student_hook)]

    def forward(self, input_ids, attention_mask, labels=None):
        return self.student(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]

        active_hook_pairs = self._active_hook_pairs()
        teacher_hooks = [teacher_hook for teacher_hook, _ in active_hook_pairs]
        student_hooks = [student_hook for _, student_hook in active_hook_pairs]
        n = len(teacher_hooks)
        obj_weights = self.train_config.objective_weights
        assert obj_weights is not None, (
            "Objective weights must be provided for training."
        )

        # total_loss = 0

        if teacher_hooks:
            assert n == len(student_hooks)

            student_output, student_cache = self.student.run_with_cache(
                input_ids,
                return_type="both",
                names_filter=lambda name: name in student_hooks,
            )

            with torch.no_grad():
                teacher_output, teacher_cache = self.teacher.run_with_cache(
                    input_ids,
                    return_type="both",
                    names_filter=lambda name: name in teacher_hooks,
                )
        else:
            student_output = self.student(input_ids, return_type="both")

            with torch.no_grad():
                teacher_output = self.teacher(input_ids, return_type="both")

        # total_loss += obj_weights.true_labels * student_output.loss
        raw_true_label_loss = student_output.loss

        # finetuning => no distillation
        if self.train_config.training_type == TrainingType.FT:
            self.log("train_loss", raw_true_label_loss, sync_dist=True)
            return raw_true_label_loss

        # distill outputs
        # total_loss += obj_weights.teacher_outputs * forward_kl_divergence(
        #     teacher_logits=teacher_output.logits,
        #     student_logits=student_output.logits,
        #     temperature=self.train_config.temperature,
        # )
        raw_divergence_loss = forward_kl_divergence(
            teacher_logits=teacher_output.logits,
            student_logits=student_output.logits,
            temperature=self.train_config.temperature,
        )

        # distill internal acts
        act_substrings_no_attn_scores = ["embed", "attn_out", "mlp", "resid"]

        # collect raw losses by objective name
        raw_objective_losses: dict[str, torch.Tensor] = {}

        for i in range(n):
            # handle attn scores as a distribution
            # if "attn_scores" in student_hooks[i]:
            #     assert "attn_scores" in teacher_hooks[i], (
            #         f"Expected teacher to be an 'attn_scores' activation, but found {teacher_hooks[i]}"
            #     )
            #     raw_objective_losses["attn_scores"] = forward_kl_divergence(
            #         teacher_logits=teacher_cache[teacher_hooks[i]],
            #         student_logits=student_cache[student_hooks[i]],
            #         temperature=self.train_config.temperature,
            #     )
            #     break

            for act in act_substrings_no_attn_scores:
                if act in student_hooks[i]:
                    assert act in teacher_hooks[i], (
                        f"Expected to find teacher's '{act}' activation, but found {teacher_hooks[i]}"
                    )

                    if self.train_config.projection:
                        projector = self._projector_for_hooks(
                            teacher_hooks[i], student_hooks[i]
                        )
                        if self.train_config.projection.direction == ProjectionDirection.UP:
                            raw_loss = self.mse(
                                teacher_cache[teacher_hooks[i]],
                                projector(student_cache[student_hooks[i]]),
                            )
                        elif self.train_config.projection.direction == ProjectionDirection.DOWN:
                            raw_loss = self.mse(
                                projector(teacher_cache[teacher_hooks[i]]),
                                student_cache[student_hooks[i]],
                            )
                        else:
                            raise ValueError(
                                f"Unrecognized projection direction: {self.train_config.projection.direction.value.upper()}"
                            )
                    else:
                        raw_loss = self.mse(
                            teacher_cache[teacher_hooks[i]],
                            student_cache[student_hooks[i]],
                        )

                    # If there is exactly one activation per objective, store it directly.
                    # If multiple layers can match the same objective, sum them.
                    if act in raw_objective_losses:
                        raw_objective_losses[act] = raw_objective_losses[act] + raw_loss
                    else:
                        raw_objective_losses[act] = raw_loss

                    break  # go back to outer loop and process next hook

        # we create a zero tensor for activations we aren't using for a given experiment
        zero = torch.zeros((), device=student_output.logits.device)
        losses = RawObjectiveLosses(
            divergence=raw_divergence_loss,
            true_labels=raw_true_label_loss,
            embed=raw_objective_losses.get("embed", zero),
            attn_out=raw_objective_losses.get("attn_out", zero),
            mlp=raw_objective_losses.get("mlp", zero),
            resid=raw_objective_losses.get("resid", zero),
        )

        self._check_finite_objectives(losses, batch_idx)

        if self._uses_dynamic_objective():
            weights = self.loss_fn.compute_weights(losses)  # type: ignore[attr-defined]
            total_loss, metrics = self.loss_fn(losses, weights)  # type: ignore[call-arg]
        else:
            total_loss, metrics = self.loss_fn(losses)

        self._check_finite_total_and_metrics(total_loss, metrics, batch_idx)

        self.log_dict(
            metrics,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )

        return total_loss

    @staticmethod
    def _check_finite_objectives(losses: RawObjectiveLosses, batch_idx: int) -> None:
        """Stop at the first invalid raw objective before weighting can hide its source."""
        for name, loss in losses.as_dict().items():
            if not torch.isfinite(loss).all():
                raise FloatingPointError(
                    f"Non-finite raw loss at batch {batch_idx}: "
                    f"{name}={loss.detach().float().item()} "
                    f"(dtype={loss.dtype}, device={loss.device})"
                )

    @staticmethod
    def _check_finite_total_and_metrics(
        total_loss: torch.Tensor,
        metrics: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Detect NaNs introduced by objective weighting or metric construction."""
        if not torch.isfinite(total_loss).all():
            raise FloatingPointError(
                f"Non-finite total loss at batch {batch_idx}: "
                f"{total_loss.detach().float().item()}"
            )

        for name, value in metrics.items():
            if not torch.isfinite(value).all():
                raise FloatingPointError(
                    f"Non-finite training metric at batch {batch_idx}: "
                    f"{name}={value.detach().float().item()}"
                )

    def on_after_backward(self) -> None:
        """Report the first parameter whose gradient becomes non-finite."""
        for name, parameter in self.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(
                    f"Non-finite gradient after backward: {name} "
                    f"(dtype={parameter.grad.dtype}, device={parameter.grad.device})"
                )
    

    def validation_step(self, batch, batch_idx):
        target_device = next(self.student.parameters()).device
        batch = batch.to(target_device)
        logits = self.student(batch["input_ids"], return_type="logits")

        # 2. Align Logits and Labels
        # TransformerLens logits are [batch, seq_len, vocab_size]
        # We need to predict the NEXT token, so we shift:

        # Shift logits: Remove the last token's prediction (it has no target)
        shift_logits = logits[..., :-1, :].contiguous()

        # Shift labels: Remove the first token (it has no history to predict it from)
        shift_labels = batch["labels"][..., 1:].contiguous()

        # 3. Calculate Cross Entropy manually
        # 'ignore_index=-100' is the magic key that prevents NaNs from padding
        val_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        perplexity = torch.exp(val_loss)

        self.log("val_loss", val_loss, prog_bar=False)
        self.log("val_ppl", perplexity, prog_bar=False)

        return val_loss

    # def on_validation_epoch_end(self):
    #     self.log_dict(self.val_metrics.compute(), sync_dist=True)
    #     self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        target_device = next(self.student.parameters()).device
        batch = batch.to(target_device)

        logits = self.student(batch["input_ids"], return_type="logits")

        # 2. Align Logits and Labels
        # TransformerLens logits are [batch, seq_len, vocab_size]
        # We need to predict the NEXT token, so we shift:

        # Shift logits: Remove the last token's prediction (it has no target)
        shift_logits = logits[..., :-1, :].contiguous()

        # Shift labels: Remove the first token (it has no history to predict it from)
        shift_labels = batch["labels"][..., 1:].contiguous()

        # 3. Calculate Cross Entropy manually
        # 'ignore_index=-100' is the magic key that prevents NaNs from padding
        test_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        perplexity = torch.exp(test_loss)

        self.log("test_loss", test_loss, prog_bar=False)
        self.log("test_ppl", perplexity, prog_bar=False)

        return test_loss

    # def on_test_epoch_end(self):
    #     self.log_dict(self.test_metrics.compute(), sync_dist=True)
    #     self.test_metrics.reset()

    def configure_optimizers(self):
        if isinstance(self.projector, (OnlineProjector, nn.ModuleDict)) and any(
            isinstance(module, OnlineProjector)
            for module in self.projector.modules()
        ):
            projector_parameters = [
                parameter
                for module in self.projector.modules()
                if isinstance(module, OnlineProjector)
                for parameter in module.parameters()
            ]
            optimizer = torch.optim.AdamW(
                [
                    {"params": self.student.parameters()},
                    {"params": projector_parameters},
                ],
                lr=self.train_config.learning_rate,
                weight_decay=self.train_config.weight_decay,
            )
        else:
            optimizer = torch.optim.AdamW(
                self.student.parameters(),
                lr=self.train_config.learning_rate,
                weight_decay=self.train_config.weight_decay,
            )
            if self.projector:
                self.projector.eval()

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
            n_samples=self.train_config.num_samples,
            streaming=self.train_config.streaming,
        )

        dataset = dataset.shuffle(seed=self.train_config.seed, buffer_size=1000)

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
            n_samples=self.train_config.num_samples,
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

    # def test_dataloader(self):
    #     dataset = load_data(
    #         dataset_name=self.train_config.dataset.name,
    #         subset=self.train_config.dataset.subset,
    #         tokenizer=self.tokenizer,
    #         seq_len=self.train_config.seq_len,
    #         split="test",
    #         n_samples=self.train_config.n_samples,
    #         streaming=self.train_config.streaming,
    #     )

    #     return DataLoader(
    #         dataset,
    #         batch_size=self.train_config.batch_size,
    #         shuffle=False,
    #         num_workers=self.train_config.num_data_workers,
    #         collate_fn=self.data_collator,
    #         pin_memory=True,
    #     )

    @classmethod
    def create(
        cls,
        cfg: ExperimentConfig,
        dirs: ExpInstDirs,
        hyperparams: HparamsConfig,
        training_params: TrainingParams,
        eval_tasks: LMEvalTasks,
    ):
        pl.seed_everything(cfg.seed, workers=True)

        if cfg.training_type == TrainingType.NO_TRAIN:
            no_train_param = -1
            no_train_config = StudentModuleConfig(
                exp_name=cfg.name,
                distill_pair=cfg.distill_pair,
                dirs=dirs,
                projection=cfg.projection,
                dataset=training_params.dataset,
                seq_len=training_params.seq_len,
                batch_size=training_params.train_batch_size,
                learning_rate=no_train_param,
                weight_decay=no_train_param,
                max_grad_norm=no_train_param,
                temperature=no_train_param,
                seed=cfg.seed,
                torch_dtype=training_params.torch_dtype,
                num_data_workers=training_params.resources.student.num_data_workers,
                total_tokens=training_params.total_tokens,
                objective_weights=None,
                objective_variant=None,
                total_steps=no_train_param,
                num_samples=no_train_param,
                finetune_teacher=training_params.controls.finetune_teacher,
                training_type=cfg.training_type,
                streaming=training_params.resources.stream_task_data,
                device=training_params.device,
            )
            return StudentModule(train_config=no_train_config)

        run_name = f"student_{cfg.name}"
        builder = load_dataset_builder(
            training_params.dataset.name, training_params.dataset.subset
        )
        n_total_train_samples = builder.info.splits["train"].num_examples  # type: ignore
        n_train_samples = (
            min(n_total_train_samples, training_params.num_samples)
            if training_params.num_samples
            else n_total_train_samples
        )
        total_steps = math.ceil(
            (n_train_samples * training_params.num_epochs)
            / training_params.train_batch_size
        )

        def train_student_per_worker(config):
            if training_params.reproducible_backend:
                ray.train.torch.enable_reproducibility()

            assert cfg.objective is not None, (
                "Objective config must be provided if training."
            )
            train_config = StudentModuleConfig(
                exp_name=cfg.name,
                distill_pair=cfg.distill_pair,
                dirs=dirs,
                projection=cfg.projection,
                dataset=training_params.dataset,
                seq_len=training_params.seq_len,
                batch_size=training_params.train_batch_size,
                learning_rate=config["learning_rate"],
                weight_decay=config["weight_decay"],
                max_grad_norm=config["max_grad_norm"],
                temperature=config["temperature"],
                seed=cfg.seed,
                torch_dtype=training_params.torch_dtype,
                num_data_workers=training_params.resources.student.num_data_workers,
                total_tokens=training_params.total_tokens,
                objective_weights=cfg.objective.weights,
                objective_variant=cfg.objective.weight_str,
                total_steps=total_steps,
                num_samples=n_train_samples,
                finetune_teacher=training_params.controls.finetune_teacher,
                training_type=cfg.training_type,
                streaming=training_params.resources.stream_task_data,
                device=training_params.device,
            )

            module = StudentModule(train_config=train_config)
            uses_cuda = training_params.uses_cuda

            if config.get("ddp", False):
                trainer = pl.Trainer(
                    devices="auto",
                    accelerator="auto" if uses_cuda else "cpu",
                    strategy=RayDDPStrategy(find_unused_parameters=True),
                    plugins=[RayLightningEnvironment()],
                    callbacks=[RayTrainReportCallback()],
                    max_epochs=training_params.num_epochs,
                    gradient_clip_algorithm="norm",
                    gradient_clip_val=config["max_grad_norm"],
                    enable_progress_bar=False,
                    enable_checkpointing=False,
                    deterministic=training_params.reproducible_backend,
                    val_check_interval=max(1.0, int(total_steps * 0.2)),
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
                    strategy=RayDDPStrategy(find_unused_parameters=True),
                    plugins=[RayLightningEnvironment()],
                    callbacks=[RayTrainReportCallback()],
                    max_epochs=training_params.num_epochs,
                    gradient_clip_algorithm="norm",
                    gradient_clip_val=config["max_grad_norm"],
                    enable_progress_bar=False,
                    enable_checkpointing=False,
                    deterministic=training_params.reproducible_backend,
                    val_check_interval=max(1.0, int(total_steps * 0.2)),
                    logger=WandbLogger(
                        name=config["wandb_run_name"],
                        project=training_params.wandb.project,
                        group=training_params.wandb.group,
                    ),
                )
            try:
                wandb.init(
                    project=training_params.wandb.project,
                    group=training_params.wandb.group,
                    name=config["wandb_run_name"],
                    reinit=True,
                )
                trainer = prepare_trainer(trainer)
                module.train()
                trainer.fit(module)

                if ray.train.get_context().get_world_rank() == 0:
                    with tempfile.TemporaryDirectory() as eval_dir:
                        eval_dir_path = Path(eval_dir)
                        module.save_pretrained(str(eval_dir_path))
                        extracted_dict, results_df, raw_results_dict = evaluate_model(
                            model_path=str(eval_dir_path),
                            tasks=eval_tasks.names,
                            experiment_name=config["wandb_run_name"],
                            seed=cfg.seed,
                            batch_size=training_params.test_batch_size,
                            limit=eval_tasks.sample_limit,
                            device=training_params.device,
                        )
                        lm_eval_mean = mean_task_score(extracted_dict)
                        if math.isnan(lm_eval_mean):
                            raise RuntimeError(
                                f"lm_eval produced no usable task scores for {config['wandb_run_name']}"
                            )

                    with tempfile.TemporaryDirectory() as artifact_dir:
                        artifact_path = Path(artifact_dir)
                        trainer.save_checkpoint(str(artifact_path / "checkpoint.ckpt"))
                        save_eval_scores(
                            scores_dir=artifact_path / "scores",
                            extracted_dict=extracted_dict,
                            results_df=results_df,
                            raw_results_dict=raw_results_dict,
                        )
                        report_metrics = {
                            "lm_eval_mean": lm_eval_mean,
                        }
                        val_ppl = trainer.callback_metrics.get("val_ppl")
                        if isinstance(val_ppl, torch.Tensor):
                            report_metrics["val_ppl"] = val_ppl.detach().float().item()
                        elif val_ppl is not None:
                            report_metrics["val_ppl"] = float(val_ppl)
                        ray.train.report(
                            report_metrics,
                            checkpoint=Checkpoint.from_directory(artifact_dir),
                        )
            finally:
                wandb.finish()

        def train_fixed_params(fixed_params: FixedHparamSet):
            visible_gpus = training_params.resources.visible_gpus if training_params.uses_cuda else 0
            role_resources = training_params.resources.student
            num_ray_workers = role_resources.resolve_num_ray_workers(
                fixed_mode=True,
                visible_gpus=visible_gpus,
            )
            num_cpus_per_trial = role_resources.resolve_num_cpus_per_trial(visible_gpus)
            fixed_config = {
                "learning_rate": fixed_params.lr,
                "weight_decay": fixed_params.weight_decay,
                "max_grad_norm": fixed_params.max_grad_norm,
                "temperature": fixed_params.temperature,
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
                component="student",
                mode="fixed",
                visible_gpus=visible_gpus,
                num_workers=num_ray_workers,
                num_gpus_per_trial=1.0 if training_params.uses_cuda else 0.0,
                num_cpus_per_trial=num_cpus_per_trial,
            )
            train_run_config = ray.train.RunConfig(
                name=f"{run_name}_fixed_{uuid.uuid4().hex[:6]}",
                checkpoint_config=ray.tune.CheckpointConfig(
                    num_to_keep=1,
                    checkpoint_frequency=0,
                    checkpoint_score_attribute="lm_eval_mean",
                    checkpoint_score_order="max",
                ),
                failure_config=ray.train.FailureConfig(
                    max_failures=training_params.resources.max_failures
                ),
                storage_path=str(dirs.ray),
            )
            trainer = ray.train.torch.TorchTrainer(
                train_student_per_worker,
                train_loop_config=fixed_config,
                run_config=train_run_config,
                scaling_config=scaling_config,
            )

            return trainer.fit()

        def tune_asha(num_trials: int, tune_params: TuneParams):
            visible_gpus = training_params.resources.visible_gpus if training_params.uses_cuda else 0
            role_resources = training_params.resources.student
            num_ray_workers = role_resources.resolve_num_ray_workers(
                fixed_mode=False,
                visible_gpus=visible_gpus,
            )
            num_cpus_per_trial = role_resources.resolve_num_cpus_per_trial(visible_gpus)
            tune_config = {
                "learning_rate": _choice_or_range(tune_params.lr, kind="qloguniform", quantization=1e-6),
                "weight_decay": _choice_or_range(tune_params.weight_decay, kind="qloguniform", quantization=1e-4),
                "max_grad_norm": _choice_or_range(tune_params.max_grad_norm, kind="uniform"),
                "temperature": _choice_or_range(tune_params.temperature, kind="uniform"),
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
                component="student",
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
                    checkpoint_score_attribute="lm_eval_mean",
                    checkpoint_score_order="max",
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
                train_student_per_worker,
                scaling_config=scaling_config,
            )

            optuna_search = OptunaSearch(
                metric="val_ppl",  # Preserving your optimization goal
                mode="min",
                seed=cfg.seed,
            )

            search_alg = ConcurrencyLimiter(
                optuna_search,
                max_concurrent=max(
                    1,
                    math.floor(
                        max(1, visible_gpus)
                        / 1.0
                    ),
                )
            )

            asha_scheduler = ASHAScheduler(
                time_attr="training_iteration",
                max_t=total_steps,
                grace_period=max(1, total_steps * 0.2),  # Minimum number of iterations before pruning can occur
                reduction_factor=3,
            )
            tuner = ray.tune.Tuner(
                trainable=ray_ptl_trainer,
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

        fixed_params = get_fixed_student_params(
            hyperparams, cfg.student_type, cfg.training_type
        )

        if fixed_params:
            best_result = train_fixed_params(fixed_params=fixed_params)
        elif hyperparams.tune:
            best_result = tune_asha(
                num_trials=training_params.n_trials, tune_params=hyperparams.tune
            ).get_best_result("lm_eval_mean", "max")
        else:
            raise ValueError("Neither valid fixed params nor tuning params were found.")

        assert best_result.checkpoint is not None, (
            "No checkpoint found in best Student result."
        )
        with best_result.checkpoint.as_directory() as checkpoint_dir:
            module = StudentModule.load_from_checkpoint(
                f"{checkpoint_dir}/checkpoint.ckpt"
            )
            scores_dir = Path(checkpoint_dir) / "scores"
            if scores_dir.exists():
                cached_scores_dir = Path(
                    tempfile.mkdtemp(prefix="kdfp_lm_eval_scores_", dir="/tmp")
                )
                shutil.copytree(scores_dir, cached_scores_dir, dirs_exist_ok=True)
                module._lm_eval_scores_dir = cached_scores_dir
            return module

    @property
    def model_file(self) -> str:
        return gen_student_weights_filename()

    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        student_tl_state_dict = self.student.state_dict()
        save_path = f"{path}/{self.model_file}"
        torch.save(student_tl_state_dict, save_path)

        student_hf_state_dict, _ = convert_hooked_to_hf_weights(
            self.student, cfg=self.student.cfg
        )
        student_name = self.train_config.distill_pair.student.name.lower()
        if "gemma-3" in student_name or "gemma3" in student_name:
            student_hf_state_dict = {
                key: value
                for key, value in student_hf_state_dict.items()
                if ".bias" not in key
            }

        if self.train_config.distill_pair.student.layer_reduction_factor:
            hf, _, _ = load_shallow(
                model_name=self.train_config.distill_pair.student.name,
                mode="hf",
                layer_reduction_factor=self.train_config.distill_pair.student.layer_reduction_factor,
                selection_method="linspace",
                torch_dtype=self.train_config.torch_dtype,
            )
        else:
            hf = AutoModelForCausalLM.from_pretrained(
                self.train_config.distill_pair.student.name,
                dtype=self.train_config.torch_dtype,
            )
        hf.load_state_dict(student_hf_state_dict)
        hf.save_pretrained(path)  # type: ignore
        self.tokenizer.save_pretrained(path)  # type: ignore

        cached_scores_dir = getattr(self, "_lm_eval_scores_dir", None)
        if cached_scores_dir is not None and Path(cached_scores_dir).exists():
            shutil.copytree(
                cached_scores_dir,
                Path(path) / "scores",
                dirs_exist_ok=True,
            )

        with open(f"{path}/pl_config.json", "w") as f:
            json.dump(self.train_config.model_dump(), f, indent=4)

        print(f"Student saved to {path}")

    @classmethod
    def load_pretrained(cls, path: str) -> "StudentModule":
        with open(f"{path}/pl_config.json", "r") as f:
            hparams = json.load(f)
        cfg = StudentModuleConfig(**hparams)
        student_module = cls(train_config=cfg)
        student_tl_state_dict = torch.load(
            Path(path) / student_module.model_file, weights_only=True
        )
        student_module.student.load_state_dict(student_tl_state_dict)
        student_module.tokenizer = AutoTokenizer.from_pretrained(path)
        student_module.student.tokenizer = student_module.tokenizer
        print(f"Student loaded from {path}")
        return student_module
