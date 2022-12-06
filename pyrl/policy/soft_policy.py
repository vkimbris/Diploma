import torch

from .base_policy import BasePolicy
from typing import Callable


class SoftPolicy(BasePolicy):
    def __init__(self,
                 q: torch.nn.Module,
                 epsilon: float = 0.2):

        super(SoftPolicy, self).__init__(q)

        self.epsilon = epsilon

    def sample(self, state: torch.tensor) -> torch.tensor:
        pass
