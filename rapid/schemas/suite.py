from copy import deepcopy
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import yaml
from pydantic import Field, field_validator, model_validator
from transformer_lens import HookedTransformer

from rapid.schemas.configBase import ConfigBase
from rapid.schemas.directories import DirectoryParser
from rapid.schemas.experiments.experimentInstance import (
    ExperimentInstance,
)
from rapid.schemas.experiments.experimentSpace import ExperimentSpace
from rapid.schemas.hparams import HparamsConfig
from rapid.schemas.models import (
    DistillationGroups,
    DistillationModelConfig,
    DistillationPair,
    ModelSelection,
)
from rapid.schemas.tasks import LMEvalTasks
from rapid.schemas.trainingParams import TrainingParams
from rapid.utils.general import get_hook_block_idx
from rapid.utils.shallow_model_loading import get_indices_and_num_layers_to_keep


class Suite(ConfigBase):
    suite_title: str
    models: ModelSelection
    directories: DirectoryParser
    experiment_space: ExperimentSpace
    training_params: TrainingParams
    hook_types: Dict[str, List[str]] = Field(
        ..., description="Keys 'single' and 'per_layer' with lists of hook names"
    )
    hparams: HparamsConfig
    # A dictionary where the key is the task name
    # (e.g., 'hellaswag') and the value is its config
    tasks: LMEvalTasks
    distill_groups: DistillationGroups = Field(
        default=None, init=False, validate_default=False
    )  # type: ignore

    # @computed_field
    # @property
    # def task_list(self) -> List[TaskInstance]:
    #     return [
    #         TaskInstance(name=name, dataset_name=cfg.dataset_name, subset=cfg.subset)
    #         for name, cfg in self.tasks.items()
    #     ]

    # validator to ensure hook point keys are expected
    @field_validator("hook_types")
    def validate_hook_keys(cls, v):
        allowed_keys = {"single", "per_layer"}
        if not all(key in allowed_keys for key in v.keys()):
            raise ValueError(f"hook_types can only contain the keys: {allowed_keys}")
        return v

    @model_validator(mode="after")
    def build_distill_groups(self) -> "Suite":
        raw_teacher_pretrained: DistillationModelConfig = self._generate_hook_list(
            model_name=self.models.teacher.name,
        )

        student_pretrained: DistillationModelConfig = self._generate_hook_list(
            model_name=self.models.pretrained.name,
            tokenizer_name=self.models.pretrained.tokenizer_name,
        )

        teacher_shallow: DistillationModelConfig = self._generate_hook_list(
            model_name=self.models.teacher.name,
            layer_reduction_factor=self.models.shallow.layer_reduction_factor,
            selection_method="linspace",
        )

        """
        - The shallow student is first n layers because once we actually load in the shallow student,
          the first n layers will be the entire student model.
        - This way, the linspaced teacher layers match up with all of the student layers
        - NOTE: for narrow students, we initialize the full hook list here and let the
                ExperimentConfig apply the kd_layer_filter_factors
        """
        student_shallow: DistillationModelConfig = self._generate_hook_list(
            model_name=self.models.shallow.name,
            layer_reduction_factor=self.models.shallow.layer_reduction_factor,
            selection_method="first_n",
        )

        assert raw_teacher_pretrained.d_model == teacher_shallow.d_model

        # need to select teacher blocks which align with the pretrained student's
        def _filter_teacher_hooks_for_pretrained_student(
            raw_teacher_pretrained: DistillationModelConfig,
            student_pretrained: DistillationModelConfig,
        ) -> DistillationModelConfig:
            """
            - compute block indices to keep
            - robust to student layers <= teacher layers
            - ensures no collisions in the block indices (i.e. full distillation across all pretrained student layers)
            """
            teacher_n_distilled_layers = min(
                raw_teacher_pretrained.n_distilled_layers,
                student_pretrained.n_distilled_layers,
            )
            selected_blocks = np.arange(teacher_n_distilled_layers)
            selected_blocks = (
                selected_blocks * (raw_teacher_pretrained.n_distilled_layers - 1)
            ) // (teacher_n_distilled_layers - 1)
            assert len(set(selected_blocks)) == len(selected_blocks)
            selected_blocks = set(selected_blocks)

            filtered_teacher_hooks = []
            for hook in raw_teacher_pretrained.hook_list:
                # add in all hooks outside of transformer blocks
                if "blocks" not in hook:
                    filtered_teacher_hooks.append(hook)
                else:
                    block_idx = get_hook_block_idx(hook)
                    if block_idx in selected_blocks:
                        filtered_teacher_hooks.append(hook)

            corrected_teacher_pretrained = deepcopy(raw_teacher_pretrained)
            corrected_teacher_pretrained.hook_list = filtered_teacher_hooks
            corrected_teacher_pretrained.n_distilled_layers = teacher_n_distilled_layers

            return corrected_teacher_pretrained

        corrected_teacher_pretrained = _filter_teacher_hooks_for_pretrained_student(
            raw_teacher_pretrained, student_pretrained
        )

        shallow_pair = DistillationPair(
            teacher=teacher_shallow, student=student_shallow
        )
        pretrained_pair = DistillationPair(
            teacher=corrected_teacher_pretrained,
            student=student_pretrained,
        )

        self.distill_groups = DistillationGroups(
            shallow=shallow_pair,
            pretrained=pretrained_pair,
        )

        return self

    # Class method to load from a YAML file
    @classmethod
    def from_yaml(cls, file_path: str) -> "Suite":
        with open(file_path, "r") as f:
            raw_config = yaml.safe_load(f)
        return cls(**raw_config)

    def _generate_hook_list(
        self,
        model_name: str,
        tokenizer_name: Optional[str] = None,
        layer_reduction_factor: Optional[float] = None,
        selection_method: Optional[Literal["linspace", "first_n"]] = None,
    ) -> DistillationModelConfig:
        lm = HookedTransformer.from_pretrained(
            model_name, device=self.training_params.device
        )
        lm_cfg = lm.cfg
        del lm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        d_model = lm_cfg.d_model
        if layer_reduction_factor is not None:
            assert selection_method, (
                "You must supply a selection method if performing layer reduction"
            )
            layer_idx_list, n_layers_to_keep = get_indices_and_num_layers_to_keep(
                total_layers=lm_cfg.n_layers,
                layer_reduction_factor=layer_reduction_factor,
                selection_method=selection_method,
            )
        else:
            layer_idx_list = list(range(lm_cfg.n_layers))
            n_layers_to_keep = lm_cfg.n_layers

        hooks = []
        hooks.extend(self.hook_types["single"])
        for idx in layer_idx_list:
            for hp in self.hook_types["per_layer"]:
                hooks.append(f"blocks.{idx}.{hp}")
        return DistillationModelConfig(
            name=model_name,
            hook_list=hooks,
            d_model=d_model,
            n_distilled_layers=n_layers_to_keep,
            layer_reduction_factor=layer_reduction_factor,
            tokenizer_name=tokenizer_name,
        )

    def generate_experiment_list(self) -> Tuple[List[ExperimentInstance], List[str]]:
        base_dirs = self.directories.resolve(suite_title=self.suite_title)

        exp_cfg_list = self.experiment_space.generate_experiment_cfgs(
            suite_title=self.suite_title,
            distill_groups=self.distill_groups,
            get_baselines=self.training_params.controls.get_baselines,
        )
        exp_inst_dirs = [base_dirs.create_instance_dirs(ep) for ep in exp_cfg_list]
        cfg_and_dirs = zip(exp_cfg_list, exp_inst_dirs)

        experiment_list = [
            ExperimentInstance(
                cfg=cfg,
                dirs=dirs,
                training_params=self.training_params,
                hparams=self.hparams,
                tasks=self.tasks,
            )
            for (cfg, dirs) in cfg_and_dirs
        ]

        print(f"Total experiments created: {len(experiment_list)}")

        unique_experiment_list = list(set(experiment_list))
        exp_names = [e.cfg.name for e in unique_experiment_list]
        sorted_experiment_list = sorted(
            unique_experiment_list, key=lambda exp: exp.cfg.name
        )

        # should not have created any duplicate experiments
        assert len(experiment_list) == len(sorted_experiment_list)

        return sorted_experiment_list, sorted(exp_names)

    # def run_baseline_teacher_tests(self) -> List[dict]:
    #     pl.seed_everything(seed=self.experiment_space.default_seed, workers=True)
    #     results_list = []

    #     def trainer_factory() -> pl.Trainer:
    #         return pl.Trainer(
    #             accelerator="gpu",
    #             strategy="auto",
    #             devices=1,
    #             deterministic=self.training_params.reproducible_backend,
    #             default_root_dir=self.directories.root
    #             + "/"
    #             + self.directories.lightning,
    #         )

    #     for task in self.task_list:
    #         base_teacher_cfg = BaselineModuleConfig(
    #             model_name=self.models.teacher.name,
    #             is_shallow=False,
    #             task=task,
    #             seq_len=self.training_params.seq_len,
    #             batch_size=self.training_params.test_batch_size,
    #             seed=self.experiment_space.default_seed,
    #             torch_dtype=self.training_params.torch_dtype,
    #             num_data_workers=self.training_params.resources.teacher.num_data_workers,
    #             n_samples=self.training_params.num_samples,
    #             layer_reduction_factor=None,
    #             stream_data=self.training_params.resources.stream_task_data,
    #         )
    #         base_teacher_module = BaselineModule(base_teacher_cfg)
    #         base_teacher_module.print_param_count("base_teacher")
    #         base_teacher_results = trainer_factory().test(
    #             model=base_teacher_module, verbose=True
    #         )
    #         results_list.append(
    #             generate_result(
    #                 name="base_teacher",
    #                 model=base_teacher_module.model,
    #                 task_name=task.name,
    #                 metric_name="accuracy",
    #                 score=base_teacher_results[0]["test_acc"],
    #             )
    #         )

    #         base_teacher_scores_dir = (
    #             self.directories.get_suite_scores_dir(self.suite_title) / "base_teacher"
    #         )
    #         base_teacher_scores_dir.mkdir(parents=True, exist_ok=True)
    #         pd.DataFrame(results_list).to_parquet(
    #             base_teacher_scores_dir / f"results_df_{get_timestamp()}.parquet"
    #         )

    #     return results_list
