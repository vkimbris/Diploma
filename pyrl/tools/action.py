import numpy as np

from typing import List, Tuple


class Action:
    def __init__(self) -> None:
        pass


class DiscreteAction(Action):
    def __init__(self, name: str, number: int = None, value: float = None) -> None:
        super().__init__()

        self.name = name
        self.number = number
        self.value = value

    def __repr__(self) -> str:
        return f'Action "{self.name}"'
    

class ContinousAction(np.ndarray, Action):

    def __new__(cls, a):
        obj = np.asarray(a).view(cls)
        return obj
