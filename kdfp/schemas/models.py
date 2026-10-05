from typing import List, Optional

from kdfp.schemas.configBase import ConfigBase
from kdfp.utils.pydantic_custom_types import FlexFloat


class DistillationModelConfig(ConfigBase):
    name: str
    d_model: int
    n_distilled_layers: int
    hook_list: List[str]
    layer_reduction_factor: Optional[float] = None
    tokenizer_name: Optional[str] = None


class DistillationPair(ConfigBase):
    teacher: DistillationModelConfig
    student: DistillationModelConfig


class DistillationGroups(ConfigBase):
    shallow: DistillationPair
    pretrained: DistillationPair


class ModelParser(ConfigBase):
    name: str
    tokenizer_name: Optional[str] = None
    layer_reduction_factor: FlexFloat = None
    kd_layer_filter_factors: List[FlexFloat] = [1]


class ModelSelection(ConfigBase):
    teacher: ModelParser
    shallow: ModelParser
    pretrained: ModelParser
