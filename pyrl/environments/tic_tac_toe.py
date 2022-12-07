import torch

from typing import Tuple

from .base_env import BaseEnvironment
from ..agents import Agent

class TicTacToe(BaseEnvironment):

    def __init__(self, player: Agent, first: bool = False):
        super(TicTacToe, self).__init__()

        self.player = player
        self.first = first

        self.field = torch.torch.full(size=(9, ), fill_value=.5)
        self.chip = 1 if self.first else 0


    def init(self) -> torch.tensor:
        if self.first:
            return self.field

        player_turn = self.player.action(self.field, torch.arange(9))

        self.field[player_turn] = 1

        return self.field

    def step(self, action: torch.tensor) -> Tuple[torch.tensor, float, bool]:
        # algorithm decision
        self.field[action] = self.chip

        if torch.all(self.field != .5):
            return self.field, self.__get_final_reward(), True

        # player decision
        player_turn = self.player.action(self.field, self.get_available_actions(self.field))
        self.field[player_turn] = 1 - self.chip

        if torch.all(self.field != .5):
            return self.field, self.__get_final_reward(), True

        return self.field, 0.0, False

    def get_available_actions(self, state: torch.tensor) -> torch.tensor:
        return torch.where(state == .5)[0]

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



