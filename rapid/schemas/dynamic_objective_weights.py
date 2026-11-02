from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


OBJECTIVE_NAMES = (
    "divergence",
    "true_labels",
    "embed",
    "attn_out",
    "mlp",
    "resid",
)


@dataclass(slots=True)
class RawObjectiveLosses:
    divergence: torch.Tensor
    true_labels: torch.Tensor
    embed: torch.Tensor
    attn_out: torch.Tensor
    mlp: torch.Tensor
    resid: torch.Tensor

    def as_tensor(self) -> torch.Tensor:
        return torch.stack(
            (
                self.divergence,
                self.true_labels,
                self.embed,
                self.attn_out,
                self.mlp,
                self.resid,
            )
        )

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "divergence": self.divergence,
            "true_labels": self.true_labels,
            "embed": self.embed,
            "attn_out": self.attn_out,
            "mlp": self.mlp,
            "resid": self.resid,
        }


@dataclass(slots=True)
class DynamicObjectiveWeights:
    divergence: float
    true_labels: float
    embed: float
    attn_out: float
    mlp: float
    resid: float

    def as_tensor(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        return torch.tensor(
            (
                self.divergence,
                self.true_labels,
                self.embed,
                self.attn_out,
                self.mlp,
                self.resid,
            ),
            device=device,
            dtype=dtype,
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "divergence": self.divergence,
            "true_labels": self.true_labels,
            "embed": self.embed,
            "attn_out": self.attn_out,
            "mlp": self.mlp,
            "resid": self.resid,
        }


class DynamicObjectiveLossPL(nn.Module):
    """
    Hybrid dynamic weighting:
      - learned uncertainty weighting
      - EMA-based progress balancing

    Intended usage in PyTorch Lightning:
      1) Build ObjectiveLosses each training step.
      2) Call compute_weights(losses).
      3) Call forward(losses, weights) to get total_loss and metrics.
      4) Log metrics with self.log_dict(...) inside training_step.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        ema_decay: float = 0.99,
        eps: float = 1e-8,
        min_weight: float = 0.05,
        max_weight: float = 0.80,
        metric_prefix: str = "train",
        active_objectives: Optional[set[str] | frozenset[str]] = None,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.ema_decay = ema_decay
        self.eps = eps
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.metric_prefix = metric_prefix
        self.active_objectives = frozenset(
            OBJECTIVE_NAMES if active_objectives is None else active_objectives
        )
        unknown = self.active_objectives - set(OBJECTIVE_NAMES)
        if unknown:
            raise ValueError(f"Unknown active objectives: {sorted(unknown)}")

        self.log_vars = nn.Parameter(torch.zeros(len(OBJECTIVE_NAMES)))
        self.register_buffer("ema_losses", torch.ones(len(OBJECTIVE_NAMES)))

    def compute_weights(self, losses: RawObjectiveLosses) -> DynamicObjectiveWeights:
        losses_tensor = losses.as_tensor()

        with torch.no_grad():
            self.ema_losses.mul_(self.ema_decay).add_(
                (1.0 - self.ema_decay) * losses_tensor.detach()
            )

        # Learned uncertainty weighting
        uncertainty_weights = torch.exp(-self.log_vars)

        # EMA-based progress weighting
            # Divergence should be nonnegative, but tiny floating-point residuals can be
        # negative after probability calculations. Fractional powers of those
        # residuals would turn the weights into NaN.
        progress_ratio = torch.clamp_min(
            losses_tensor.detach() / (self.ema_losses + self.eps), 0.0
        )
        progress_weights = progress_ratio.pow(self.alpha)

        # Combine the two signals
        active_mask = torch.tensor(
            [name in self.active_objectives for name in OBJECTIVE_NAMES],
            device=losses_tensor.device,
            dtype=torch.bool,
        )
        raw_weights = uncertainty_weights * progress_weights
        raw_weights = raw_weights.masked_fill(~active_mask, 0.0)
        weights_tensor = raw_weights / (raw_weights.sum() + self.eps)

        # Stabilize
        weights_tensor = torch.where(
            active_mask,
            torch.clamp(weights_tensor, self.min_weight, self.max_weight),
            torch.zeros_like(weights_tensor),
        )
        weights_tensor = weights_tensor / (weights_tensor.sum() + self.eps)

        return DynamicObjectiveWeights(
            divergence=float(weights_tensor[0].detach()),
            true_labels=float(weights_tensor[1].detach()),
            embed=float(weights_tensor[2].detach()),
            attn_out=float(weights_tensor[3].detach()),
            mlp=float(weights_tensor[4].detach()),
            resid=float(weights_tensor[5].detach()),
        )

    def forward(
        self,
        losses: RawObjectiveLosses,
        weights: DynamicObjectiveWeights,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        losses_tensor = losses.as_tensor()
        weights_tensor = weights.as_tensor(
            device=losses_tensor.device,
            dtype=losses_tensor.dtype,
        )

        weighted_losses_tensor = weights_tensor * losses_tensor
        total_loss = weighted_losses_tensor.sum()

        # Uncertainty regularization
        active_mask = torch.tensor(
            [name in self.active_objectives for name in OBJECTIVE_NAMES],
            device=losses_tensor.device,
            dtype=torch.bool,
        )
        total_loss = total_loss + self.log_vars.masked_select(active_mask).sum()

        metrics: dict[str, torch.Tensor] = {}
        raw_losses_dict = losses.as_dict()
        weights_dict = weights.as_dict()

        for idx, name in enumerate(OBJECTIVE_NAMES):
            metrics[f"{self.metric_prefix}/loss_raw/{name}"] = raw_losses_dict[name].detach()
            metrics[f"{self.metric_prefix}/loss_weighted/{name}"] = weighted_losses_tensor[idx].detach()
            metrics[f"{self.metric_prefix}/weight/{name}"] = torch.tensor(
                weights_dict[name],
                device=losses_tensor.device,
                dtype=losses_tensor.dtype,
            )

        metrics[f"{self.metric_prefix}/loss_total"] = total_loss.detach()

        return total_loss, metrics


class StaticObjectiveLossPL(nn.Module):
    """Sum active raw objectives without dynamic reweighting."""

    def __init__(
        self,
        active_objectives: Optional[set[str] | frozenset[str]] = None,
        metric_prefix: str = "train",
    ) -> None:
        super().__init__()
        self.metric_prefix = metric_prefix
        self.active_objectives = frozenset(
            OBJECTIVE_NAMES if active_objectives is None else active_objectives
        )
        unknown = self.active_objectives - set(OBJECTIVE_NAMES)
        if unknown:
            raise ValueError(f"Unknown active objectives: {sorted(unknown)}")

    def forward(
        self,
        losses: RawObjectiveLosses,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        losses_dict = losses.as_dict()
        weights = {
            name: float(name in self.active_objectives)
            for name in OBJECTIVE_NAMES
        }
        total_loss = sum(
            (losses_dict[name] for name in self.active_objectives),
            torch.zeros_like(losses.divergence),
        )

        metrics: dict[str, torch.Tensor] = {}
        for name in OBJECTIVE_NAMES:
            raw_loss = losses_dict[name]
            metrics[f"{self.metric_prefix}/loss_raw/{name}"] = raw_loss.detach()
            metrics[f"{self.metric_prefix}/loss_weighted/{name}"] = (
                raw_loss * weights[name]
            ).detach()
            metrics[f"{self.metric_prefix}/weight/{name}"] = torch.tensor(
                weights[name], device=raw_loss.device, dtype=raw_loss.dtype
            )

        metrics[f"{self.metric_prefix}/loss_total"] = total_loss.detach()
        return total_loss, metrics
    

# ========== EXAMPLE USAGE ================
# def training_step(self, batch, batch_idx):
#     divergence_loss, true_labels_loss, embed_loss, attn_out_loss, mlp_loss, resid_loss = self.compute_objectives(batch)

#     losses = RawObjectiveLosses(
#         divergence=divergence_loss,
#         true_labels=true_labels_loss,
#         embed=embed_loss,
#         attn_out=attn_out_loss,
#         mlp=mlp_loss,
#         resid=resid_loss,
#     )

#     weights = self.loss_fn.compute_weights(losses)
#     total_loss, metrics = self.loss_fn(losses, weights)

#     self.log_dict(
#         metrics,
#         on_step=True,
#         on_epoch=False,
#         prog_bar=False,
#         logger=True,
#     )

#     return total_loss
