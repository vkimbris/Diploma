import torch
import tkinter as tk

from ..policy import BasePolicy

from typing import Tuple


class BaseEnvironment:
    def __init__(self,
                 action_space: torch.tensor,
                 state_dim: int,
                 vparams: dict = None):

        self.action_space = action_space
        self.state_dim = state_dim

        if vparams is not None:
            self._app = tk.Tk()
            self._app.geometry(f"{vparams['width']}x{vparams['height']}")

    def init(self) -> torch.tensor:
        pass

    def step(self, action: torch.tensor) -> Tuple[torch.tensor, float, bool]:
        pass

    def get_available_actions(self, state: torch.tensor) -> torch.tensor:
        pass
    
    def visualize(self, policy: BasePolicy):
        self._app.mainloop()
