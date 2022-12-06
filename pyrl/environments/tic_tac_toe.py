import torch
import tkinter as tk

from torch import Tensor

from .base_env import BaseEnvironment
from ..policy import BasePolicy
from typing import Tuple


class TicTacToe(BaseEnvironment):

    def __init__(self, player: BasePolicy, first: bool = False, vparams: dict = None):
        super(TicTacToe, self).__init__(state_dim=9,
                                        action_space=torch.arange(9),
                                        vparams=vparams)

        self.player = player
        self.first = first

        self.field = torch.torch.full(size=(self.state_dim, ), fill_value=.5)
        self.chip = 1 if self.first else 0

        self.available_actions = action_space

    def init(self) -> torch.tensor:
        if self.first:
            return self.field

        player_turn = self.player.sample(self.field)

        self.field[player_turn] = 1

        return self.field

    def step(self, action: torch.tensor) -> Tuple[torch.tensor, float, bool]:
        # algorithm decision
        self.field[action.item()] = self.chip

        if torch.all(self.field != .5):
            return self.field, self.__get_final_reward(), True

        # player decision
        player_turn = self.player.sample(self.field)
        self.field[player_turn] = 1 - self.chip

        if torch.all(self.field != .5):
            return self.field, self.__get_final_reward(), True

        return self.field, 0.0, False

    def get_available_actions(self, state: torch.tensor) -> torch.tensor:
        pass

    def visualize(self, policy: BasePolicy):
        # some code to visualize Tic Tac Toe

        super(TicTacToe, self).visualize(policy)

    def __get_final_reward(self) -> torch.tensor:
        field = self.field.reshape(3, 3)

        main_diag = torch.diag(field, 0)
        sub_diag = torch.diag(torch.rot90(field), 0)

        for chip in [0, 1]:
            if torch.all(field[0, :] == chip) or torch.all(field[1, :] == chip) or torch.all(field[2, :] == chip):
                return torch.tensor(2 * chip - 1)

            if torch.all(field[:, 0] == chip) or torch.all(field[:, 1] == chip) or torch.all(field[:, 2] == chip):
                return torch.tensor(2 * chip - 1)

            if torch.all(main_diag == chip) or torch.all(sub_diag == chip):
                return torch.tensor(2 * chip - 1)

        return torch.tensor(0)



