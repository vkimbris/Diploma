import torch

from .base_alg import BaseAlgorithm
from typing import Any

class MonteCarlo(BaseAlgorithm):
    def __init__(self, network: torch.nn.Module, gamma: float, optimizer: Any, loss_func: Any):
        super(MonteCarlo, self).__init__(network, gamma, optimizer, loss_func)

    def get_target(self,
                   actions: torch.tensor, 
                   new_states: torch.tensor, 
                   rewards: torch.tensor) -> torch.tensor:
        
        pass
