from .policy import BasePolicy, GreedyPolicy
from .environments import BaseEnvironment
from .algorithms.value.base_alg import BaseAlgorithm

from tqdm import tqdm


class Experiment:
    def __init__(self, algorithm: BaseAlgorithm, policy: BasePolicy, environment: BaseEnvironment):
        self.algorithm = algorithm
        self.policy = policy
        self.environment = environment

        self.memory = []

    def run(self, n_epochs: int, progress=False):
        state = self.environment.init()

        for _ in tqdm(range(n_epochs), disable=not progress):
            available_actions = self.environment.get_available_actions(state)

            action = self.policy.sample(state)
            new_state, reward, terminal = self.environment.step(action)

            if terminal:
                return

            state = new_state

            self.memory.append((state, action, reward, new_state))

    def train(self) -> None:
        self.algorithm.train(self.memory)

        self.policy.update(self.algorithm.q)

    def get_greedy_policy(self) -> GreedyPolicy:
        return GreedyPolicy(self.policy.q)
