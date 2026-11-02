import torch
import torch.nn.functional as F


def forward_kl_divergence(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    temperature: float,
    mask_value: float = -1e8,
) -> torch.Tensor:
    """
    Forward KL divergence loss for knowledge distillation with causal masking
    support. The teacher distribution is the reference distribution.

    This implementation handles:
    - Masked positions in attention logits (e.g., -inf values).
    - Temperature scaling for softening distributions.
    - Proper T² compensation for correct gradient scaling in distillation.
    - Numerical stability for all calculations.

    Args:
        teacher_logits (torch.Tensor): Logits from the true distribution (P).
        student_logits (torch.Tensor): Logits from the approximating distribution (Q).
        temperature (float): Temperature for softening. Must be positive.
        mask_value (float): Value to replace -inf with for numerical stability.

    Returns:
        torch.Tensor: A single scalar loss value with proper temperature compensation.
    """
    # Step 0:   Cut off dead output rows
    #           - Pythia 6.9b and 12b extend their output dim for GPU efficiency,
    #              but still only use the same 50,277 tokens as the rest of the suite
    teacher_logits = teacher_logits[:, :, :student_logits.shape[-1]]

    # Step 1: Input Validation
    assert temperature > 0, "Temperature must be positive."
    assert teacher_logits.shape == student_logits.shape, (
        "Logits must have the same shape. Found teacher: {}, student: {}".format(
            teacher_logits.shape, student_logits.shape
        )
    )

    # Step 2: Detect masked positions BEFORE any modifications
    # A position is masked if it's -inf or a large negative number in either tensor.
    p_mask = torch.isinf(teacher_logits) | (teacher_logits <= mask_value * 0.9)
    q_mask = torch.isinf(student_logits) | (student_logits <= mask_value * 0.9)
    combined_mask = p_mask | q_mask

    # Step 3: Clean logits and apply temperature scaling
    p_clean = torch.where(torch.isinf(teacher_logits), mask_value, teacher_logits)
    q_clean = torch.where(torch.isinf(student_logits), mask_value, student_logits)

    scaled_p = p_clean / temperature
    scaled_q = q_clean / temperature

    # Step 4: Compute probabilities and log-probabilities
    # Using log_softmax is more numerically stable than softmax followed by log.
    p_log_probs = F.log_softmax(scaled_p, dim=-1)
    q_log_probs = F.log_softmax(scaled_q, dim=-1)

    p_probs = torch.exp(p_log_probs)
    q_probs = torch.exp(q_log_probs)

    # Forward KL: KL(P || Q) = sum(P * (log(P) - log(Q))).
    kl_elements = p_probs * (p_log_probs - q_log_probs)

    # Zero out contributions from masked positions.
    kl_elements = kl_elements.masked_fill(combined_mask, 0.0)

    # Compute the mean loss over all unmasked elements.
    num_unmasked = (~combined_mask).sum()

    if num_unmasked == 0:
        # Handle the edge case where every position is masked
        return torch.tensor(0.0, device=teacher_logits.device, dtype=kl_elements.dtype)

    # The sum is over all dimensions (batch, sequence, vocab)
    loss = kl_elements.sum() / num_unmasked

    # Apply T² scaling for knowledge distillation.
    # This ensures the gradients are correctly scaled with respect to temperature.
    loss = loss * (temperature**2)

    return loss
