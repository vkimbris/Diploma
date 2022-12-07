import torch

from torch.distributions import Multinomial

from ..algorithms.value import BaseAlgorithm

from typing import Callable


class Agent:
    def __init__(self, algorithm: BaseAlgorithm):
        self.algorithm = algorithm

    def action(self, state: torch.tensor, available_actions, epsilon: float) -> torch.tensor:
        greedy_action = self.greedy_action(state, available_actions)

        # do greedy acion
        if torch.rand(1) > epsilon:
            return greedy_action
        # do random non greedy action
        else:
            return available_actions[torch.randint(0, len(available_actions), (1,))[0]]

    
    def greedy_action(self, state: torch.tensor, available_actions: torch.tensor = None) -> torch.tensor:        
        output = self.algorithm.network(state)
        
        # forbid to choose not available actions
        output[~torch.isin(torch.arange(len(output)), available_actions)] = -torch.inf

        return torch.argmax(output)

    def __call__(self, state: torch.tensor) -> torch.tensor:
        return self.algorithm.network(state)



    
