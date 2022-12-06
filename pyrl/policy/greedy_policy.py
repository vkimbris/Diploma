import torch

from .base_policy import BasePolicy


class GreedyPolicy(BasePolicy):
    def __init__(self, q: torch.nn.Module):
        super(GreedyPolicy, self).__init__(q)

    def sample(self, state: torch.tensor) -> torch.tensor:
        return torch.argmax(self.q(state))

