from enum import Enum


class ProjectionTraining(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    RANDOMIZED = "randomized"


class StudentType(str, Enum):
    SHALLOW = "shallow"
    PRETRAINED = "pretrained"


class ProjectionDirection(str, Enum):
    UP = "up"
    DOWN = "down"


def safe_enum(enum_class, value):
    """Convert a value to an enum member, or return None if value is None."""
    if value is None:
        return None
    return enum_class(value)


class ObjectiveWeightStr(str, Enum):
    ONLY_OUTPUTS = "only_outputs"
    BALANCED = "balanced"
    FINETUNING = "finetuning"

    # "No" variants
    NO_EMBED = "no_embed"
    # NO_ATTN_SCORES = "no_attn_scores"
    NO_ATTN_OUT = "no_attn_out"
    NO_MLP = "no_mlp"
    NO_RESID = "no_resid"

    # "With" variants
    WITH_EMBED = "with_embed"
    # WITH_ATTN_SCORES = "with_attn_scores"
    WITH_ATTN_OUT = "with_attn_out"
    WITH_MLP = "with_mlp"
    WITH_RESID = "with_resid"


class TrainingType(str, Enum):
    KD = "kd"
    FT = "ft"
    NO_TRAIN = "no_train"
