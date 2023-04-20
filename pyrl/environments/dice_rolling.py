import random

from typing import Tuple, List

from ..tools import *
from .env import Environment


class DiceRolling(Environment):

    def __init__(self) -> None:

        action_space = [DiscreteAction("Бросить"), DiscreteAction("Не бросать")]
        state_space = [DiscreteState(f"S{k}") for k in range(7)] + [DiscreteState("Терминальное состояние", True)]

        super().__init__(action_space, state_space)


    def init(self) -> DiscreteState:
        return self.state_space[0]
    
    def step(self, state: DiscreteState, action: DiscreteAction) -> Tuple[DiscreteState, float]:
        if action.name == "Бросить" and state.name == "S0":
            return random.choice(self.state_space[1:7]), 0
        
        if action.name == "Бросить" and state.name != "S0":
            return self.state_space[-1], random.choice([1, 2, 3, 4, 5, 6])
        
        if action.name == "Не бросать":
            return self.state_space[-1], int(state.name[-1])
    
    def get_available_actions(self, state: DiscreteState) -> List[DiscreteAction]:
        if state.name == "S0":
            return [self.action_space[0]]
        
        else:
            return self.action_space
