from typing import Optional, Union

from pydantic import model_validator

from rapid.schemas.configBase import ConfigBase
from rapid.utils.enums import StudentType, TrainingType


class MiniMaxiHyperparam(ConfigBase):
    mini: Optional[Union[float, int]] = None
    maxi: Optional[Union[float, int]] = None
    choices: Optional[list[Union[float, int]]] = None

    @model_validator(mode="after")
    def validate_search_space(self) -> "MiniMaxiHyperparam":
        if self.choices is not None:
            if not self.choices:
                raise ValueError("choices must not be empty")
            return self

        if self.mini is None or self.maxi is None:
            raise ValueError("Either choices or both mini/maxi must be provided")

        return self


class TuneParams(ConfigBase):
    lr: MiniMaxiHyperparam
    weight_decay: MiniMaxiHyperparam
    temperature: MiniMaxiHyperparam
    max_grad_norm: MiniMaxiHyperparam


def _choice_or_range(spec: MiniMaxiHyperparam, *, kind: str, quantization: float | None = None):
    from ray import tune

    if spec.choices is not None:
        return tune.choice(spec.choices)

    if spec.mini is None or spec.maxi is None:
        raise ValueError("Search space spec must define either choices or mini/maxi")

    if kind == "qloguniform":
        if quantization is None:
            raise ValueError("qloguniform requires a quantization value")
        return tune.qloguniform(spec.mini, spec.maxi, quantization)
    if kind == "quniform":
        if quantization is None:
            raise ValueError("quniform requires a quantization value")
        return tune.quniform(spec.mini, spec.maxi, quantization)
    if kind == "uniform":
        return tune.uniform(spec.mini, spec.maxi)

    raise ValueError(f"Unsupported tune kind: {kind}")


class FixedHparamSet(ConfigBase):
    lr: float
    weight_decay: float
    max_grad_norm: float
    temperature: Optional[float] = None


class DistillPairHparams(ConfigBase):
    ft: Optional[FixedHparamSet] = None
    kd: Optional[FixedHparamSet] = None


class FixedOffProjHparamSet(ConfigBase):
    lr: float
    weight_decay: float
    tied_weights: bool


class FixedParams(ConfigBase):
    teacher: Optional[FixedHparamSet] = None
    pretrained: Optional[DistillPairHparams] = None
    shallow: Optional[DistillPairHparams] = None
    off_proj: Optional[FixedOffProjHparamSet] = None


class HparamsConfig(ConfigBase):
    fixed: Optional[FixedParams] = None
    tune: Optional[TuneParams] = None


def get_fixed_student_params(
    hyperparams: HparamsConfig, student_type: StudentType, training_type: TrainingType
):
    if not hyperparams.fixed:
        return None

    # Map types to the specific config objects
    try:
        student_group = {
            StudentType.SHALLOW: hyperparams.fixed.shallow,
            StudentType.PRETRAINED: hyperparams.fixed.pretrained,
        }.get(student_type)

        if not student_group:
            return None

        return {
            TrainingType.FT: student_group.ft,
            TrainingType.KD: student_group.kd,
        }.get(training_type)
    except AttributeError:
        return None
