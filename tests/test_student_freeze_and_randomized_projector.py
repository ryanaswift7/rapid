import torch
import torch.nn as nn

from rapid.core.projector.randomized.RandomizedProjector import RandomizedProjector
from rapid.core.student.student import StudentModule
from rapid.utils.enums import ProjectionDirection


def test_student_train_keeps_teacher_frozen_and_in_eval_mode():
    module = StudentModule.__new__(StudentModule)
    nn.Module.__init__(module)
    module.teacher = nn.BatchNorm1d(4)
    module.teacher.requires_grad_(True)

    module.train()

    assert not module.teacher.training
    assert all(not parameter.requires_grad for parameter in module.teacher.parameters())


def test_randomized_projector_supports_bfloat16_and_has_independent_weights():
    first = RandomizedProjector(
        teacher_dim=8,
        student_dim=4,
        direction=ProjectionDirection.DOWN,
        torch_dtype=torch.bfloat16,
    )
    second = RandomizedProjector(
        teacher_dim=8,
        student_dim=4,
        direction=ProjectionDirection.DOWN,
        torch_dtype=torch.bfloat16,
    )

    assert first.projection_matrix.shape == (8, 4)
    assert first.projection_matrix.dtype == torch.bfloat16
    assert torch.equal(first.projection_matrix.abs(), second.projection_matrix.abs())
    assert not torch.equal(first.projection_matrix, second.projection_matrix)
    assert list(first.parameters()) == []


def test_randomized_projector_up_direction_has_expected_shape():
    projector = RandomizedProjector(
        teacher_dim=8,
        student_dim=4,
        direction=ProjectionDirection.UP,
        torch_dtype=torch.float32,
    )

    assert projector.projection_matrix.shape == (4, 8)
    assert projector(torch.randn(2, 4)).shape == (2, 8)
