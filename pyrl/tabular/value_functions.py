import itertools
import random

import numpy as np

from typing import Tuple, List, Callable

from tqdm import tqdm

from ..tools import DiscreteAction, DiscreteState


class Q:

    def __init__(self, state_space: List[DiscreteState], action_space: List[DiscreteAction]) -> None:
        self.state_space = state_space
        self.action_space = action_space
        
        self.values = {pair: random.random() for pair in itertools.product(*[state_space, action_space])}

    def __call__(self, state: DiscreteState, action: DiscreteAction) -> float:
        return self.values[state, action]
    

class V:
    pass
