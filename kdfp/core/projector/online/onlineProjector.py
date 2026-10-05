import torch.nn as nn
from kdfp.utils.enums import ProjectionDirection
import torch

class OnlineProjector(nn.Module):
    def __init__(self,
                 teacher_dim: int,
                 student_dim: int, 
                 direction: ProjectionDirection,
                 torch_dtype: torch.dtype):
        super(OnlineProjector, self).__init__()
        self.teacher_dim = teacher_dim
        self.student_dum = student_dim
        self.direction = direction
        self.torch_dtype = torch_dtype

        if direction == ProjectionDirection.UP:
            self.project = nn.Linear(student_dim, teacher_dim, bias=False, dtype=torch_dtype)
        else:
            self.project = nn.Linear(teacher_dim, student_dim, bias=False, dtype=torch_dtype)

    def forward(self, x):
        return self.project(x)