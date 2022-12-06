import torch


class BaseAlgorithm:
    def __init__(self, q: torch.nn.Module):
        self.q = q

    def update(self, batch: torch.Tensor) -> None:
        pass