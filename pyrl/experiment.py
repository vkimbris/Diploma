from .environments import BaseEnvironment
from .agents import Agent

from tqdm import tqdm


class Experiment:
    def __init__(self, agent: Agent, environment: BaseEnvironment):
        self.agent = agent
        self.environment = environment

        self.memory = []

    def explore(self, n_epochs: int, progress=False):
        state = self.environment.init()

        for _ in tqdm(range(n_epochs), disable=not progress):
            print(state)
            available_actions = self.environment.get_available_actions(state)

            action = self.agent.action(state, available_actions, epsilon=0.2)
            new_state, reward, terminal = self.environment.step(action)

            self.memory.append((state, action, reward, new_state))

            state = new_state

            if terminal:
                return self.memory

        return self.memory

    def __train_alg(self) -> None:
        self.agent.algorithm.train(self.memory)

