from typing import Tuple, List

from ..tools import *


class Environment:
    
    def __init__(self, action_space=None, state_space=None) -> None:
        self.action_space = action_space
        self.state_space = state_space

    def init(self) -> State:
        pass

    def step(self, state: State, action: Action) -> Tuple[State, float]:
        pass

    def get_available_actions(self, state: State) -> List[Action]:
        pass

    def spaces(self) -> Tuple:
        return self.state_space, self.action_space
