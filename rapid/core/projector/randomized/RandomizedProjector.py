import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rapid.utils.enums import ProjectionDirection


class RandomizedProjector(nn.Module):
    def __init__(
        self,
        teacher_dim: int,
        student_dim: int,
        direction: ProjectionDirection,
        torch_dtype: torch.dtype,
    ):
        super(RandomizedProjector, self).__init__()
        self.teacher_dim = teacher_dim
        self.student_dim = student_dim
        self.direction = direction
        self.torch_dtype = torch_dtype

        """
        Here I'm using the binary coin implementation of the JL Lemma from
        Achlioptas (2003), which:
        - creates a projection matrix P of size (student_dim, teacher_dim)
        - each entry P_ij chosen independently with:
            - P_ij = +1 with probability 1/2
            - P_ij = -1 with probability 1/2
        - the final P is scaled by 1/sqrt(k)

        NOTE: JL Lemma only applies to dimension reduction (i.e. down direction)
        Even though there are no theoretical guarantees, the upward direction
        (scaled by 1/sqrt(teacher_dim) instead of 1/sqrt(student_dim)) is tested
        to evaluate performance empirically
        """
        p = torch.randint(0, 2, (teacher_dim, student_dim), dtype=torch.int8)
        p = (2 * p - 1).to(dtype=torch_dtype) / math.sqrt(student_dim)

        if direction == ProjectionDirection.UP:
            scaling_adjustment = math.sqrt(student_dim / teacher_dim)
            p = p.T * scaling_adjustment

        self.register_buffer("projection_matrix", p)

    def forward(self, x):
        # F.linear has an implicit transpose, so I transpose it back
        # I could instead create a transposed matrix, but I find this more intuitive
        return F.linear(x, self.projection_matrix.T)  # type: ignore
