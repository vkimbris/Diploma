import torch
import torch.nn as nn

from pyrl.environments import TicTacToe
from pyrl.policy import SoftPolicy, GreedyPolicy
from pyrl.experiment import Experiment
from pyrl.algorithms.value import MonteCarlo


class Q(nn.Module):
    def __init__(self):
        super(Q, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(9, 18),
            nn.Sigmoid(),
            nn.Linear(18, 9)
        )

    def forward(self, x):
        return self.linear(x)


q = Q()

player = GreedyPolicy(q=q)
env = TicTacToe(player=player)


s0 = env.init()
a0 = torch.tensor(3)

s1 = env.step(a0)

print(s1)