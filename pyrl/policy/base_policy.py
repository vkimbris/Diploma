import torch

from typing import Callable


class BasePolicy:
    def __init__(self, q: torch.nn.Module):
        self.q = q
        self.action_actions = available_actions

    def sample(self, state: torch.tensor) -> torch.tensor:
        pass

    def update(self, q: torch.nn.Module) -> None:
        pass


