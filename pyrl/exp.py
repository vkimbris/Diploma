import itertools

from .agents import Agent
from .environments import Environment

from tqdm import tqdm


class Experiment:

    def __init__(self, environment: Environment) -> None:
        self.environment = environment

    def run(self, agent: Agent, max_iter: int = 100) -> None:
        state = self.environment.init()
        
        history = []

        iter = 0
        while not state.terminal and iter != max_iter:
            action = agent.action(state, self.environment.get_available_actions(state))
            
            new_state, reward = self.environment.step(state, action)
            
            history.append((
                state, action, reward, new_state
            ))

            if new_state.terminal:
                break

            state = new_state

            iter += 1

        return history
    
    def explore(self, agent: Agent, n_times: int, max_iter: int = 100, for_training: bool = True) -> list:
        result = []

        for _ in tqdm(range(n_times)):
            result.append(self.run(agent, max_iter))

        if for_training:
            return list(itertools.chain.from_iterable(result))

        return result



    

    