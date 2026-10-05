from pathlib import Path
from typing import Any

import torch
from pydantic import field_serializer, field_validator, model_serializer
from transformer_lens import HookedTransformer


class DtypeSerializerMixin:
    """A mixin to provide serialization for a 'dtype' field."""

    @field_validator("torch_dtype", mode="before")
    def validate_dtype(cls, v):
        if isinstance(v, str):
            v = getattr(torch, v)
        return v

    @field_serializer("torch_dtype")
    def serialize_dtype(self, torch_dtype: torch.dtype) -> str:
        # Converts torch.float32 -> "float32"
        return str(torch_dtype).split(".")[-1]


class MCEvalMixin:
    def _shared_eval_step(self, model, batch, batch_idx):
        # The batch contains tensors of shape [batch_size, num_task_choices, seq_len]
        batch_size, num_choices, seq_len = batch["input_ids"].shape
        true_labels = batch["label"]

        # 1. Reshape inputs for a single, efficient forward pass
        # Shape becomes [batch_size * num_task_choices, seq_len]
        input_ids = batch["input_ids"].view(-1, seq_len)
        # attention_mask = batch["attention_mask"].view(-1, seq_len)

        # 2. Perform one forward pass for all choices in the batch
        with torch.no_grad():
            if isinstance(model, HookedTransformer):
                logits = model(input_ids)
            else:
                outputs = model(input_ids)
                logits = outputs.logits

        # --- Add the debugging check for safety ---
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            raise ValueError("NaN or Inf in logits, stopping validation.")

        # 3. Calculate the Negative Log-Likelihood (NLL) for each choice
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        # Use reduction='none' to get the loss for each token individually
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

        nll_per_token = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        # Reshape the per-token losses to match the sequence structure
        nll_per_token = nll_per_token.view(batch_size * num_choices, -1)

        # Ignore the loss for padding tokens by setting them to zero
        nll_per_token[shift_labels == self.tokenizer.pad_token_id] = 0  # type: ignore

        # Sum the NLL for each sequence to get a score per choice
        nll_scores = nll_per_token.sum(dim=1)

        # 4. Reshape scores and find the prediction
        # Reshape scores from [batch_size * 4] to [batch_size, 4]
        choice_nll_scores = nll_scores.view(batch_size, num_choices)

        # The prediction is the choice with the minimum NLL
        preds = torch.argmin(choice_nll_scores, dim=1)
        correct_choice_loss = choice_nll_scores[
            torch.arange(batch_size), true_labels
        ].mean()
        return preds, true_labels, correct_choice_loss


class PathSerializerMixin:
    @model_serializer(mode="wrap")
    def serialize_paths(self, handler: Any) -> dict[str, Any]:
        """
        Intersects the model dump and converts any Path objects to strings.
        """
        # Get the standard dict from Pydantic's default handler
        data = handler(self)

        # Recursively convert Path objects to strings
        return self._convert_paths(data)

    def _convert_paths(self, obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {k: self._convert_paths(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._convert_paths(i) for i in obj]
        return obj
