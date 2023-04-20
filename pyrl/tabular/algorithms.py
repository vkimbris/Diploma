import numpy as np

from typing import List, Tuple, Callable

from tqdm import tqdm

from .value_functions import Q, V
from ..tools import DiscreteState, DiscreteAction


def q_learning(q: Q, data: List[Tuple[DiscreteState, DiscreteAction, float, DiscreteState]], lr: float = 1e-3, gamma: float = 1.0) -> Tuple[Q, Callable]:
    for transition in data:
        state, action, reward, new_state = transition

        # make without .values
        q.values[state, action] += lr * (reward + gamma * max([q.values[new_state, a] for a in q.action_space]) - q.values[state, action])

    def policy(state: DiscreteState, available_actions: List[DiscreteAction]) -> DiscreteAction:
        return available_actions[np.argmax([q.values[state, action] for action in available_actions])]

    return q, policy