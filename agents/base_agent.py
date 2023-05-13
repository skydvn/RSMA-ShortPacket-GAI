from GAI.LICE import *
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from utils.result_utils import *

class base_agent:
    def __init__(
            self,
            args,
            env,
            alg
    ):
        self.env = env
        self.model = alg

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        """
            state / state_next: matrix of 
        """
        state_next, reward, done, info = self.env.step(action, self.total_step)
        # print(f"reward: {reward}")
        if not self.is_test:
            self.transition += [reward, state_next, done]
            # print(self.transition)
            self.memory.store(*self.transition)

        return state_next, reward, done, info

    def train(self):
        pass

    def evaluate(self):
        pass

    def _plot(
            self,
            frame_idx: int,
            scores: List[float],
            actor_losses: List[float],
            critic_losses: List[float],
    ):
        """Plot the training progresses."""

        def subplot(loc: int, title: str, values: List[float]):
            plt.subplot(loc)
            plt.title(title)
            plt.plot(values)

        subplot_params = [
            (131, f"frame {frame_idx}. score: {np.mean(scores[-10:])}", scores),
            (132, "actor_loss", actor_losses),
            (133, "critic_loss", critic_losses),
        ]

        plt.figure(figsize=(30, 5))
        for loc, title, values in subplot_params:
            subplot(loc, title, values)
        plt.savefig(fname="result.pdf")
        plt.show()