import numpy as np

from typing import List


class State:
    
    def __init__(self, terminal: bool = False) -> None:
        self.terminal = terminal


class DiscreteState(State):
    
    def __init__(self, name: str, terminal: bool = False) -> None:
        super().__init__(terminal)

        self.name = name

    def __repr__(self) -> str:
        return self.name


class ContinousState(np.ndarray):

    def __new__(cls, a, terminal=False):
        obj = np.asarray(a).view(cls)
        obj.terminal = terminal
        return obj
