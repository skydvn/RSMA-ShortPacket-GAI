import h5py
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from utils.result_utils import *
from agents.modules.buffer import *
from agents.modules.utils import *
from agents.GAI.LICE import *
from agents.

class base_agent(agent_utils):
    def __init__(
            self,
            args,
            env,
            alg
    ):
        self.obs_dim = args.user_num*args.antenna_num
        self.env = env
        # self.model = VAE()      # alg
        self.max_episode = args.max_episode
        self.max_step    = args.max_step
        self.memory_size = args.memory_size
        self.batch_size  = args.batch_size
        self.memory = ReplayBuffer(self.obs_dim,self.memory_size,self.batch_size)
        self.transition = list()
        """     Agent     """
        self.plotting_interval = args.plotting_interval
        self.save_flag = args.save_flag
        self.algo_name = args.algo
        """     Agent     """
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
         )

    def select_action(self, state:np.ndarray)->np.ndarray:
        """Select action from input state"""
        selected_action = np.array([0])
        self.transition = [state,selected_action]
        return selected_action

    def step(self):
        """Take an action and return the response of the env."""
        """
            state / state_next: matrix of 
        """
        state_next, reward, done, info = self.env.step(self.total_step)

        if not self.is_test:
            self.transition += [reward, state_next, done]
            print(self.transition)
            self.memory.store(*self.transition)

        return state_next, reward, done, info

    def data_collect(self):
        num_ep = self.max_episode
        num_frames = self.max_step
        self.total_step = 0
        """      Train the agent      """
        for self.episode in range(1, num_ep + 1):
            self.is_test = False

            state = self.env.reset()

            for step in range(1, num_frames + 1):
                """ get channel in terms of state """
                selected_action = self.select_action(state)
                state_next, reward, done, info = self.step()

        """       Get data from memory buffer      """
        data_list = []
        batch_max = self.memory_size//self.batch_size
        for batch_idx in range(batch_max):
            big_batch = self.memory.sample_batch()
            data_list.append(big_batch["obs"])
        data_stack = np.stack(data_list, axis=0)

        save_data("./dataset", data_stack)

    def train(self):
        # Get data from dataset
        pass
        # Train-loader
        # Loops over train-loader

    def instant_train(self):
        plotting_interval = self.plotting_interval
        num_ep = self.max_episode
        num_frames = self.max_step
        self.total_step = 0
        """     Log initialization    """
        g_losses = []

        """      Train the agent      """
        for self.episode in range(1, num_ep + 1):
            self.is_test = False

            state = self.env.reset()

            for step in range(1, num_frames + 1):
                """ get channel in terms of state """
                selected_action = self.select_action(state)
                state_next, reward, done, info = self.step()

                # if training is ready
                if (
                    len(self.memory)>=self.batch_size
                ):
                    g_loss = self.update_model()
                    g_losses.append(g_loss)

                # plotting
                if self.total_step % plotting_interval==0:
                    self._plot(
                        self.total_step,
                        g_losses,
                        [],
                        [],
                    )
                    pass
        if self.save_flag:
            save_results(
                g_losses,
                [],
                [],
                [],
                self.algo_name
            )



    def evaluate(self):
        pass

    def _plot(
            self,
            frame_idx: int,
            g_losses: List[float],
            actor_losses: List[float],
            critic_losses: List[float],
    ):
        """Plot the training progresses."""

        def subplot(loc: int, title: str, values: List[float]):
            plt.subplot(loc)
            plt.title(title)
            plt.plot(values)

        subplot_params = [
            (131, f"frame {frame_idx}. score: {np.mean(g_losses[-10:])}", g_losses),
            (132, "actor_loss", actor_losses),
            (133, "critic_loss", critic_losses),
        ]

        plt.figure(figsize=(30, 5))
        for loc, title, values in subplot_params:
            subplot(loc, title, values)
        plt.savefig(fname="result.pdf")
        plt.show()

