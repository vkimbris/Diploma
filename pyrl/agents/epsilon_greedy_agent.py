import torch

from .agent import Agent
from ..algorithms.value.base_alg import BaseAlgorithm


class EpsilonGreedyAgent(Agent):
    def __init__(self, algorithm: BaseAlgorithm, epsilon: float):
        super().__init__(algorithm)

        self.epsilon = epsilon

    def action(self, state: torch.tensor, available_actions, epsilon: float = None) -> torch.tensor:
        return super().action(state, available_actions, self.epsilon)