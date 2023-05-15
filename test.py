import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from pyrl.environments import DiceRolling, BetDiceRolling
from pyrl.agents import Agent
from pyrl.exp import Experiment

from pyrl.tools import State, Action, DiscreteState, DiscreteAction
from pyrl.tabular import Q
from pyrl.tabular.algorithms import q_learning

from tqdm.notebook import tqdm


from typing import List, Callable


def random_policy(state: DiscreteState, available_actions: List[DiscreteAction]) -> DiscreteAction:
    return random.choice(available_actions)

def epsilon_greedy(policy: Callable[[State, List[Action]], Action], epsilon: float) -> Callable:
    def policy(state: DiscreteState, available_actions: List[DiscreteAction]) -> DiscreteAction:
        action = policy(state, available_actions)

        if random.random() <= epsilon:
            return random.choice(available_actions)
        
        return action
    
    return policy


environment = DiceRolling()

exp = Experiment(environment)

q = Q(*environment.spaces())


N = 1000

e0 = 0.99
eN = 0.01

cf = np.log(e0 / eN) / N

E = lambda n: e0 * np.exp(-cf * n)


agent = Agent(epsilon_greedy(q.to_policy(), E(0)))

rewards = []
for n in tqdm(range(1000)):
    # run episode 
    history = exp.run(agent)

    rewards.append(np.sum([tr[2] for tr in history]))

    # update values of Q function
    q = q_learning(q, history)

    # change agent's policy
    agent.change_policy(epsilon_greedy(q.to_policy(), E(n)))

rewards = np.split(np.array(rewards), 10)
rewards = list(map(np.mean, rewards))