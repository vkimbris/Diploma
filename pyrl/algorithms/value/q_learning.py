import torch
import copy

from .base_alg import BaseAlgorithm

from typing import Any


class QLearning(BaseAlgorithm):
    def __init__(self, network: torch.nn.Module, gamma: float, optimizer: Any, loss_func: Any) -> None:
        super(QLearning, self).__init__(network, gamma, optimizer, loss_func)

        self.target_network = copy.deepcopy(self.network)

    def get_target(self,
                   actions: torch.tensor, 
                   new_states: torch.tensor, 
                   rewards: torch.tensor) -> torch.tensor:
        
        return rewards + self.gamma * self.target_network(new_states).max(dim=1)[0].unsqueeze(1)

    def trigger_func(self):
        self.target_network.load_state_dict(self.network.state_dict())
