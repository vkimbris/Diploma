import torch
import random

from .environments import BaseEnvironment
from .agents import Agent

from tqdm import tqdm


class Experiment:
    def __init__(self, agent: Agent, environment: BaseEnvironment):
        self.agent = agent
        self.environment = environment

        self.memory = []
        self.rewards = []

    def explore(self,
                params: dict,
                progress=False,
                visualize: bool = False):

        n_epochs = params["n_epochs"]
        n_iter = params["n_iter"]
        batch_size = params["batch_size"]

        eps_start, eps_end = params["eps_start"], params["eps_end"]

        n_target_train = params["n_target_train"]

        Q = torch.log(torch.tensor(eps_start / eps_end)) / n_epochs
        
        for epoch in tqdm(range(n_epochs), disable=not progress):
            self.run_episode(n_iter, visualize, epsilon=eps_start * torch.e ** (-Q * epoch))

            if len(self.memory) > batch_size:
                self.__train_alg(random.choices(self.memory, k=batch_size))

            if epoch != 0 and epoch % n_target_train == 0:
                self.agent.algorithm.trigger_func()

        return self.agent, list(map(torch.mean, torch.split(torch.tensor(self.rewards), n_epochs)))

    def run_episode(self, n_iter, visualize: bool = False, epsilon: float = 0.2):
        state = self.environment.init()

        for k in range(n_iter):            
            available_actions = self.environment.get_available_actions(state)

            action = self.agent.action(state, available_actions, epsilon)
            new_state, reward, terminal = self.environment.step(action)

            self.rewards.append(reward.item())
            self.memory.append((state, action, reward, new_state))

            if visualize:
                print("State: ")
                self.environment.visualize(state)
                print("New state: ")
                self.environment.visualize(new_state)

            state = new_state

            if terminal:
                break

    def __train_alg(self, batch) -> None:
        self.agent.algorithm.train(batch)
