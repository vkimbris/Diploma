import torch

from .base_alg import BaseAlgorithm


class MonteCarlo(BaseAlgorithm):
    def __init__(self, q: torch.nn.Module):
        super(MonteCarlo, self).__init__(q)

    def train(self, batch: torch.tensor):
        pass
