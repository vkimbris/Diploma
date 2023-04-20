import torch
import torch.nn as nn

from torch.optim import Adam
from torch.nn import MSELoss

from ..tools import ContinousState, DiscreteAction

from typing import Tuple, Callable, List


def deep_q_learning(states: torch.tensor, 
                    actions: torch.tensor, 
                    rewards: torch.tensor, 
                    new_states: torch.tensor,
                    gamma: float = 0.99) -> torch.float:
    
    pass
