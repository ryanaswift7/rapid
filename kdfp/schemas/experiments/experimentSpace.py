import itertools
from typing import List

from pydantic import computed_field

from kdfp.schemas.configBase import ConfigBase
from kdfp.schemas.experiments.experimentConfig import (
    ExperimentConfig,
    ObjectiveInstance,
)
from kdfp.schemas.experiments.experimentFilter import ExperimentFilters
from kdfp.schemas.models import DistillationGroups
from kdfp.schemas.projections import ProjectionInstance, ProjectionParser
from kdfp.utils.enums import (
    ObjectiveWeightStr,
    ProjectionDirection,
    ProjectionTraining,
    StudentType,
    TrainingType,
)


class ObjectiveParams(ConfigBase):
    weight_variants: List[ObjectiveWeightStr]


class ExperimentSpace(ConfigBase):
    seeds: List[int]
    student_types: List[StudentType]
    objective: ObjectiveParams
    projection: ProjectionParser
    filters: ExperimentFilters

    @computed_field
    @property
    def default_seed(self) -> int:
        return self.seeds[0]

    def generate_experiment_cfgs(
        self, suite_title: str, distill_groups: DistillationGroups, get_baselines: List[str]
    ) -> List[ExperimentConfig]:
        kd_list = self.generate_kd_cfgs(
            suite_title=suite_title, distill_groups=distill_groups
        )
        baseline_list = (
            self.generate_baseline_cfgs(
                suite_title=suite_title,
                distill_groups=distill_groups,
                get_baselines=get_baselines,
            )
            if get_baselines
            else []
        )
        return self._apply_filter(kd_list + baseline_list)

    def generate_kd_cfgs(
        self, suite_title: str, distill_groups: DistillationGroups
    ) -> List[ExperimentConfig]:
        """
        Generates, names, and filters the suite of experiments.
        """
        candidates: List[ExperimentConfig] = []

        for seed, student in itertools.product(self.seeds, self.student_types):
            if student == StudentType.SHALLOW:
                for obj in self.objective.weight_variants:
                    candidates.append(
                        ExperimentConfig(
                            suite_title=suite_title,
                            seed=seed,
                            student_type=student,
                            distill_pair=getattr(distill_groups, student.value),
                            objective=ObjectiveInstance(
                                weight_str=obj,
                            ),
                            training_type=TrainingType.KD,
                        )
                    )

            else:
                for direction, training in itertools.product(
                    self.projection.directions, self.projection.training_methods
                ):
                    betas = (
                        self.projection.offline_alignment_betas
                        if training == ProjectionTraining.OFFLINE
                        else [None]
                    )

                    for beta, obj in itertools.product(
                        betas, self.objective.weight_variants
                    ):
                        candidates.append(
                            ExperimentConfig(
                                suite_title=suite_title,
                                seed=seed,
                                student_type=student,
                                distill_pair=getattr(distill_groups, student.value),
                                projection=ProjectionInstance(
                                    direction=direction,
                                    training_method=training,
                                    offline_alignment_beta=beta,
                                    init_mode=self.projection.init_mode,
                                ),
                                objective=ObjectiveInstance(
                                    weight_str=obj,
                                ),
                                training_type=TrainingType.KD,
                            )
                        )

        return candidates

    def generate_baseline_cfgs(
        self,
        suite_title: str,
        distill_groups: DistillationGroups,
        get_baselines: List[str],
    ) -> List[ExperimentConfig]:
        """
        Generates experiments to test all selected student types with finetuning and with no training,
        as well as testing the teacher model.
        """
        candidates: List[ExperimentConfig] = []
        enabled_baselines = set(get_baselines)

        for seed, student in itertools.product(self.seeds, self.student_types):
            if "ft" in enabled_baselines:
                candidates.append(
                    ExperimentConfig(
                        suite_title=suite_title,
                        seed=seed,
                        student_type=student,
                        distill_pair=getattr(distill_groups, student.value),
                        objective=ObjectiveInstance(
                            weight_str=ObjectiveWeightStr.FINETUNING,
                        ),
                        training_type=TrainingType.FT,
                    )
                )

            if "no_train" in enabled_baselines:
                candidates.append(
                    ExperimentConfig(
                        suite_title=suite_title,
                        seed=seed,
                        student_type=student,
                        distill_pair=getattr(distill_groups, student.value),
                        training_type=TrainingType.NO_TRAIN,
                    )
                )

            if student == StudentType.PRETRAINED:
                if "kd_traditional_fwd_kld" in enabled_baselines:
                    candidates.append(
                        ExperimentConfig(
                            suite_title=suite_title,
                            seed=seed,
                            student_type=student,
                            distill_pair=getattr(distill_groups, student.value),
                            projection=ProjectionInstance(
                                direction=ProjectionDirection.UP,
                                training_method=ProjectionTraining.ONLINE,
                            ),
                            objective=ObjectiveInstance(
                                weight_str=ObjectiveWeightStr.ONLY_OUTPUTS,
                            ),
                            training_type=TrainingType.KD,
                        )
                    )

                if "kd_resid_fwd_kld" in enabled_baselines:
                    candidates.append(
                        ExperimentConfig(
                            suite_title=suite_title,
                            seed=seed,
                            student_type=student,
                            distill_pair=getattr(distill_groups, student.value),
                            projection=ProjectionInstance(
                                direction=ProjectionDirection.UP,
                                training_method=ProjectionTraining.ONLINE,
                            ),
                            objective=ObjectiveInstance(
                                weight_str=ObjectiveWeightStr.WITH_RESID,
                            ),
                            training_type=TrainingType.KD,
                        )
                    )

        if "teacher" in enabled_baselines:
            # teacher evaluation is handled separately in ExperimentInstance.test_teacher
            # so we do not add per-experiment teacher configs here.
            pass

        # # then make no train (using default seed)
        # for student in self.student_types:
        #     candidates.append(
        #         ExperimentConfig(
        #             suite_title=suite_title,
        #             seed=self.default_seed,
        #             student_type=student,
        #             distill_pair=getattr(distill_groups, student.value),
        #             training_type=TrainingType.NO_TRAIN,
        #         )
        #     )

        return candidates

    def _apply_filter(
        self, experiments: List[ExperimentConfig]
    ) -> List[ExperimentConfig]:
        """Filters experiments based on name matching."""
        filtered = []

        include_list = self.filters.include or []
        exclude_list = self.filters.exclude or []

        for exp in experiments:
            name = exp.name

            # Handle Inclusion: If list exists, name must match at least one entry
            if include_list:
                if not any(match in name for match in include_list):
                    continue

            # Handle Exclusion: If list exists, name must NOT match any entry
            if exclude_list:
                if any(match in name for match in exclude_list):
                    continue

            filtered.append(exp)

        return filtered
