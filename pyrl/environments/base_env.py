import torch
import tkinter as tk


from typing import Tuple


class BaseEnvironment:

    def __init__(self):
        pass

    def init(self) -> torch.tensor:
        pass

    def step(self, action: torch.tensor) -> Tuple[torch.tensor, float, bool]:
        pass

    def get_available_actions(self, state: torch.tensor) -> torch.tensor:
        pass

