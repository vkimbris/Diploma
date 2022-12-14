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
        self.__init__(self.player, self.first)
        
        if self.first:
            return torch.clone(self.field)

        player_turn = self.player.action(self.field, torch.arange(9), 0)

        self.field[player_turn] = 1

        return torch.clone(self.field)

    def step(self, action: torch.tensor) -> Tuple[torch.tensor, torch.tensor, bool]:
        # algorithm decision
        self.field[action] = self.chip

        reward, terminal = self.__get_reward()

        if terminal:
            return torch.clone(self.field), reward, True

        # player decision
        player_turn = self.player.action(self.field, self.get_available_actions(self.field), 0.6)
        self.field[player_turn] = 1 - self.chip

        reward, terminal = self.__get_reward()

        if terminal:
            return torch.clone(self.field), reward, True

        return torch.clone(self.field), torch.tensor(0.0), False

    def get_available_actions(self, state: torch.tensor) -> torch.tensor:
        return torch.where(state == .5)[0]

    def visualize(self, state: torch.tensor) -> None:
        print(self.__state_to_str_field(state))

    def __get_reward(self) -> Tuple[torch.tensor, bool]:
        field = self.field.reshape(3, 3)

        main_diag = torch.diag(field, 0)
        sub_diag = torch.diag(torch.rot90(field), 0)

        for chip in [0, 1]:
            if torch.all(field[0, :] == chip) or torch.all(field[1, :] == chip) or torch.all(field[2, :] == chip):
                return torch.tensor(2 * chip - 1), True

            if torch.all(field[:, 0] == chip) or torch.all(field[:, 1] == chip) or torch.all(field[:, 2] == chip):
                return torch.tensor(2 * chip - 1), True

            if torch.all(main_diag == chip) or torch.all(sub_diag == chip):
                return torch.tensor(2 * chip - 1), True

            if torch.all(field != .5):
                return torch.tensor(-0.2), True

        return torch.tensor(0), False      

    def __state_to_str_field(self, state: torch.tensor) -> str:
        def parse(element: float) -> str:
            if element == 0.5:
                return "_"
            elif element == 1.0:
                return "X"
            elif element == 0.0:
                return "O"
            else:
                raise Exception("Undefined element.")
        
        field = state.reshape(3, 3).tolist()
        field = list(map(lambda v: list(map(parse, v)), field))

        output = ""
        for k in range(3):
            output += " ".join(field[k]) + "\n"

        return output
