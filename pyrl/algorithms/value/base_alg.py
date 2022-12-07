import torch

from typing import Any, List


class BaseAlgorithm:

    def __init__(self, network: torch.nn.Module, gamma: float, optimizer: Any, loss_func: Any):
        self.network = network
        self.gamma = gamma
        self.optimizer = optimizer
        self.loss_func = loss_func

    def train(self, batch: torch.tensor) -> None:
        states, actions, new_states, rewards = BaseAlgorithm.unpack_batch(batch)
        
        target = self.get_target(actions, new_states, rewards)

        loss = self.loss_func(target - self.network(states))
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()
        
    @staticmethod
    def unpack_batch(batch: List[tuple]) -> Any:
        states = torch.tensor([transition[0] for transition in batch]).reshape(-1, 1)
        actions = torch.tensor([transition[1] for transition in batch]).reshape(-1, 1)
        new_states = torch.tensor([transition[2] for transition in batch]).reshape(-1, 1)
        rewards = torch.tensor([transition[3] for transition in batch]).reshape(-1, 1)

        return states, actions, new_states, rewards
        
    def get_target(self, 
                   actions: torch.tensor, 
                   new_states: torch.tensor, 
                   rewards: torch.tensor) -> torch.tensor:

        pass
    
