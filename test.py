import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from torch.optim import Adam
from torch.nn import MSELoss

from pyrl.agents import Agent, EpsilonGreedyAgent
from pyrl.environments import TicTacToe
from pyrl.algorithms.value import QLearning

from pyrl.experiment import Experiment


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(9, 32),
            nn.Sigmoid(),
            nn.Linear(32, 9),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.linear(x)

dqn = DQN()

# create algorithm
q_learning = QLearning(network=dqn, 
                       gamma=1, 
                       optimizer=Adam(params=dqn.parameters(), lr=1e-2),
                       loss_func=MSELoss())

agent = Agent(q_learning)

env = TicTacToe(player=agent, first=True)

exp = Experiment(agent=agent, environment=env)


agent, rewards = exp.explore(n_epochs=50_000, progress=True, visualize=False)

plt.plot(rewards)
plt.show()


exp.explore(n_epochs=10, progress=False, visualize=True, EPS_START=0)
