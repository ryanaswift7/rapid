import torch
import pytest

from rapid.schemas.dynamic_objective_weights import (
    DynamicObjectiveLossPL,
    RawObjectiveLosses,
    StaticObjectiveLossPL,
)
from rapid.schemas.objective_weights import OBJECTIVE_WEIGHT_VARIANTS_MAP
from rapid.core.student.training_utils import forward_kl_divergence


def test_objective_variants_expose_active_objectives():
    with_resid = OBJECTIVE_WEIGHT_VARIANTS_MAP["with_resid"]
    no_resid = OBJECTIVE_WEIGHT_VARIANTS_MAP["no_resid"]

    assert with_resid.active_objectives == frozenset(
        {"divergence", "true_labels", "resid"}
    )
    assert no_resid.active_objectives == frozenset(
        {"divergence", "true_labels", "embed", "attn_out", "mlp"}
    )


def test_dynamic_weights_zero_inactive_objectives():
    losses = RawObjectiveLosses(
        divergence=torch.tensor(1.0),
        true_labels=torch.tensor(1.0),
        embed=torch.tensor(1.0),
        attn_out=torch.tensor(1.0),
        mlp=torch.tensor(1.0),
        resid=torch.tensor(1.0),
    )
    loss_fn = DynamicObjectiveLossPL(
        active_objectives=OBJECTIVE_WEIGHT_VARIANTS_MAP["with_resid"].active_objectives
    )

    weights = loss_fn.compute_weights(losses)

    assert weights.divergence > 0
    assert weights.true_labels > 0
    assert weights.resid > 0
    assert weights.embed == 0
    assert weights.attn_out == 0
    assert weights.mlp == 0
    assert sum(weights.as_dict().values()) == pytest.approx(1.0)


def test_only_outputs_has_no_activation_objectives():
    weights = OBJECTIVE_WEIGHT_VARIANTS_MAP["only_outputs"]

    assert weights.active_objectives == frozenset({"divergence", "true_labels"})


def test_static_objective_loss_sums_active_raw_losses_with_unit_weights():
    losses = RawObjectiveLosses(
        divergence=torch.tensor(2.0),
        true_labels=torch.tensor(3.0),
        embed=torch.tensor(5.0),
        attn_out=torch.tensor(7.0),
        mlp=torch.tensor(11.0),
        resid=torch.tensor(13.0),
    )
    loss_fn = StaticObjectiveLossPL(
        active_objectives={"divergence", "true_labels", "resid"}
    )

    total_loss, metrics = loss_fn(losses)

    assert total_loss == pytest.approx(18.0)
    assert metrics["train/weight/divergence"] == 1.0
    assert metrics["train/weight/true_labels"] == 1.0
    assert metrics["train/weight/resid"] == 1.0
    assert metrics["train/weight/embed"] == 0.0
    assert metrics["train/loss_weighted/embed"] == 0.0


def test_static_only_outputs_ignores_activation_losses():
    losses = RawObjectiveLosses(
        divergence=torch.tensor(2.0),
        true_labels=torch.tensor(3.0),
        embed=torch.tensor(100.0),
        attn_out=torch.tensor(100.0),
        mlp=torch.tensor(100.0),
        resid=torch.tensor(100.0),
    )

    total_loss, _ = StaticObjectiveLossPL(
        active_objectives=OBJECTIVE_WEIGHT_VARIANTS_MAP["only_outputs"].active_objectives
    )(losses)

    assert total_loss == pytest.approx(5.0)


def test_forward_kl_divergence_stays_finite_for_tiny_probabilities():
    logits = torch.tensor([[[0.0, -100.0, -200.0]]], dtype=torch.float32)

    loss = forward_kl_divergence(
        teacher_logits=logits,
        student_logits=logits,
        temperature=1.0,
    )

    assert torch.isfinite(loss)
    assert loss >= 0


def test_dynamic_weights_ignore_tiny_negative_progress_residuals():
    losses = RawObjectiveLosses(
        divergence=torch.tensor(-1e-8),
        true_labels=torch.tensor(1.0),
        embed=torch.tensor(0.0),
        attn_out=torch.tensor(0.0),
        mlp=torch.tensor(0.0),
        resid=torch.tensor(0.0),
    )
    loss_fn = DynamicObjectiveLossPL(active_objectives={"divergence", "true_labels"})

    weights = loss_fn.compute_weights(losses)

    assert all(torch.isfinite(torch.tensor(value)) for value in weights.as_dict().values())
