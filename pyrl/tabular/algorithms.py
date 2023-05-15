import numpy as np

from typing import List, Tuple, Callable

from tqdm import tqdm

from .value_functions import Q, V
from ..tools import DiscreteState, DiscreteAction


def q_learning(q: Q, data: List[Tuple[DiscreteState, DiscreteAction, float, DiscreteState]], lr: float = 1e-3, gamma: float = 1.0) -> Q:
    for transition in data:
        state, action, reward, new_state = transition

        # make without .values
        q.values[state, action] += lr * (reward + gamma * max([q.values[new_state, a] for a in q.action_space]) - q.values[state, action])

    return q


def mc_control(q: Q, data: List[Tuple[DiscreteState, DiscreteAction, float, DiscreteState]], gamma: float = 1.0) -> Q:
    G = 0
    for transition in data[::-1]:
        state, action, reward = transition[:3]
        
        G = gamma * G + reward

        n = q.occurrence[state, action]
        
        q.values[state, action] = n * q.values[state, action] / (n + 1) + G / (n + 1)

        q.occurrence[state, action] += 1

    return q