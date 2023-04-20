import random

from typing import Tuple, List

from ..tools import ContinousState, DiscreteAction
from .env import Environment


class BetDiceRolling(Environment):

    def __init__(self, balance: int) -> None:        
        super().__init__([DiscreteAction(f"Bet '{k}'", k - 1, k) for k in range(2, 7)])

        self.start_balance = balance
        self.current_balance = balance

    def init(self) -> ContinousState:
        self.current_balance = self.start_balance

        return ContinousState([self.start_balance])
    
    def step(self, state: ContinousState, action: DiscreteAction) -> Tuple[ContinousState, float]:
        win = random.choice([2, 3, 4, 5, 6])

        if state[0] <= 1:
            return ContinousState([0], True), -100

        return ContinousState([state[0] - action.value + win]), win - action.value
    
    def get_available_actions(self, state: ContinousState) -> List[DiscreteAction]:
        current_balance = state[0]

        if current_balance >= 6:
            return self.action_space
        
        else:
            return [action for action in self.action_space if action.value <= current_balance]


    