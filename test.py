import torch
import torch.nn as nn

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
            nn.Linear(9, 18),
            nn.Sigmoid(),
            nn.Linear(18, 9)
        )

    def forward(self, x):
        return self.linear(x)

dqn = DQN()

# create algorithm
q_learning = QLearning(network=dqn, 
                       gamma=1, 
                       optimizer=Adam(params=dqn.parameters(), lr=1e-3),
                       loss_func=MSELoss())

agent = EpsilonGreedyAgent(q_learning, epsilon=1)
player = EpsilonGreedyAgent(q_learning, epsilon=1)


env = TicTacToe(player=player, first=True)

exp = Experiment(agent=agent, environment=env)


memory = exp.explore(n_epochs=100, progress=False)

print(memory)