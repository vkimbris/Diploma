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
        
        self.values, self.occurrence = {}, {}

        for pair in itertools.product(*[state_space, action_space]):
            self.values[pair] = random.random()
            self.occurrence[pair] = 0

    def __call__(self, state: DiscreteState, action: DiscreteAction) -> float:
        return self.values[state, action]
    
    def to_policy(self) ->  Callable[[DiscreteState, List[DiscreteAction]], DiscreteAction]:
        
        def policy(state: DiscreteState, available_actions: List[DiscreteAction]) -> DiscreteAction:
            return available_actions[np.argmax([self.values[state, action] for action in available_actions])]
        
        return policy
    

class V:
    pass



