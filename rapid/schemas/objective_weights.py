import math

from pydantic import Field, model_validator

from rapid.schemas.configBase import ConfigBase


class ObjectiveWeights(ConfigBase):
    teacher_outputs: float = Field(
        ..., description="Weight for teacher outputs objective", ge=0
    )
    true_labels: float = Field(
        ..., description="Weight for true labels objective", ge=0
    )
    embed: float = Field(..., description="Weight for embedding layer objective", ge=0)
    attn_out: float = Field(
        ..., description="Weight for attention output objective", ge=0
    )
    mlp: float = Field(..., description="Weight for MLP layer objective", ge=0)
    resid: float = Field(
        ..., description="Weight for residual connection objective", ge=0
    )

    @property
    def active_objectives(self) -> frozenset[str]:
        """Return objective names enabled by this variant.

        The configured values select which objectives participate; dynamic
        weighting determines their runtime weights.
        """
        return frozenset(
            name
            for name in (
                "divergence",
                "true_labels",
                "embed",
                "attn_out",
                "mlp",
                "resid",
            )
            if getattr(self, "teacher_outputs" if name == "divergence" else name) > 0
        )

    @model_validator(mode="after")
    def check_weights_sum(self) -> "ObjectiveWeights":
        total = sum(
            [
                self.teacher_outputs,
                self.true_labels,
                self.embed,
                self.attn_out,
                self.mlp,
                self.resid,
            ]
        )

        # Check if sum is approximately 1.0, accounting for floating point errors
        if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(
                f"Weights must sum to 1.0, but got {total:.12f}. "
                f"Difference: {abs(total - 1.0):.2e}. "
                "Consider normalizing your weights."
            )

        return self


OBJECTIVE_WEIGHT_VARIANTS_MAP = {
    "only_outputs": ObjectiveWeights(
        **{
            "teacher_outputs": 1 / 2,
            "true_labels": 1 / 2,
            "embed": 0.0,
            "attn_out": 0.0,
            "mlp": 0.0,
            "resid": 0.0,
        }
    ),
    "balanced": ObjectiveWeights(
        **{
            "teacher_outputs": 1 / 6,
            "true_labels": 1 / 6,
            "embed": 1 / 6,
            "attn_out": 1 / 6,
            "mlp": 1 / 6,
            "resid": 1 / 6,
        }
    ),
    "no_embed": ObjectiveWeights(
        **{
            "teacher_outputs": 1 / 5,
            "true_labels": 1 / 5,
            "embed": 0.0,
            "attn_out": 1 / 5,
            "mlp": 1 / 5,
            "resid": 1 / 5,
        }
    ),
    "no_attn_out": ObjectiveWeights(
        **{
            "teacher_outputs": 1 / 5,
            "true_labels": 1 / 5,
            "embed": 1 / 5,
            "attn_out": 0.0,
            "mlp": 1 / 5,
            "resid": 1 / 5,
        }
    ),
    "no_mlp": ObjectiveWeights(
        **{
            "teacher_outputs": 1 / 5,
            "true_labels": 1 / 5,
            "embed": 1 / 5,
            "attn_out": 1 / 5,
            "mlp": 0.0,
            "resid": 1 / 5,
        }
    ),
    "no_resid": ObjectiveWeights(
        **{
            "teacher_outputs": 1 / 5,
            "true_labels": 1 / 5,
            "embed": 1 / 5,
            "attn_out": 1 / 5,
            "mlp": 1 / 5,
            "resid": 0.0,
        }
    ),
    "with_embed": ObjectiveWeights(
        **{
            "teacher_outputs": 1 / 3,
            "true_labels": 1 / 3,
            "embed": 1 / 3,
            "attn_out": 0.0,
            "mlp": 0.0,
            "resid": 0.0,
        }
    ),
    "with_attn_out": ObjectiveWeights(
        **{
            "teacher_outputs": 1 / 3,
            "true_labels": 1 / 3,
            "embed": 0.0,
            "attn_out": 1 / 3,
            "mlp": 0.0,
            "resid": 0.0,
        }
    ),
    "with_mlp": ObjectiveWeights(
        **{
            "teacher_outputs": 1 / 3,
            "true_labels": 1 / 3,
            "embed": 0.0,
            "attn_out": 0.0,
            "mlp": 1 / 3,
            "resid": 0.0,
        }
    ),
    "with_resid": ObjectiveWeights(
        **{
            "teacher_outputs": 1 / 3,
            "true_labels": 1 / 3,
            "embed": 0.0,
            "attn_out": 0.0,
            "mlp": 0.0,
            "resid": 1 / 3,
        }
    ),
    "finetuning": ObjectiveWeights(
        **{
            "teacher_outputs": 0.0,
            "true_labels": 1.0,
            "embed": 0.0,
            "attn_out": 0.0,
            "mlp": 0.0,
            "resid": 0.0,
        }
    ),
}
