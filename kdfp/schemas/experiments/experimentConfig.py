from typing import Optional

from pydantic import computed_field

from kdfp.schemas.configBase import ConfigBase
from kdfp.schemas.models import DistillationPair
from kdfp.schemas.objective_weights import (
    OBJECTIVE_WEIGHT_VARIANTS_MAP,
    ObjectiveWeights,
)
from kdfp.schemas.projections import ProjectionInstance
from kdfp.utils.enums import (
    ObjectiveWeightStr,
    ProjectionTraining,
    StudentType,
    TrainingType,
)


class ObjectiveInstance(ConfigBase):
    weight_str: ObjectiveWeightStr

    @computed_field
    @property
    def weights(self) -> ObjectiveWeights:
        return OBJECTIVE_WEIGHT_VARIANTS_MAP[self.weight_str]


class ExperimentConfig(ConfigBase):
    suite_title: str
    seed: int
    student_type: StudentType
    distill_pair: DistillationPair
    objective: Optional[ObjectiveInstance] = None
    projection: Optional[ProjectionInstance] = None
    training_type: TrainingType

    @computed_field
    @property
    def name(self) -> str:
        parts = [self.student_type.value, self.training_type.value]

        """
            Walrus Operator (:=) creates a local variable proj while
            checking for existence
        """
        if proj := self.projection:
            meth = proj.training_method
            proj_slug = f"{proj.direction.value}_{meth.value}"
            proj_slug += f"_{proj.init_mode}"

            if meth == ProjectionTraining.OFFLINE:
                proj_slug += f"_align_{proj.offline_alignment_beta}"

            parts.append(proj_slug)

        if obj := self.objective:
            if self.training_type == TrainingType.KD:
                parts.append(obj.weight_str.value)

        return "-".join(parts)
