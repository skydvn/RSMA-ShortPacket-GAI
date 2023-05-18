import h5py
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from utils.result_utils import *
from agents.modules.buffer import *
from agents.modules.utils import *
from agents.GAI.LICE import *
from agents.GAI.GAN import *
from agents.GAI.VAE import *
from agents.GAI.AE import *

class base_agent(agent_utils):
    def __init__(
            self,
            args,
            env,
            alg
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
         )
        self.obs_dim = args.user_num*args.antenna_num
        self.env = env
        # self.model = VAE()      # alg
        self.max_episode = args.max_episode
        self.max_step    = args.max_step
        self.memory_size = args.memory_size
        self.batch_size  = args.batch_size
        self.memory = ReplayBuffer(self.obs_dim, 1, self.memory_size,self.batch_size)
        self.transition = list()
        """     Agent     """
        self.plot_interval = args.plot_interval
        self.save_flag = args.save_flag
        self.algo_name = args.algo
        """     Agent     """
        self.generative_model = VAE(
            NaiveEncoder(input_dim=args.user_num * args.antenna_num,
                         hidden_dim=int(args.user_num * args.antenna_num / 1),
                         latent_dim=int(args.user_num * args.antenna_num / 2)),
            NaiveDecoder(latent_dim=int(args.user_num * args.antenna_num / 2),
                         hidden_dim=int(args.user_num * args.antenna_num / 1),
                         output_dim=args.user_num * args.antenna_num),
            args.beta, args.capacity, args.capacity_leadin
        ).to(self.device)
        self.generative_opt = optim.Adam(self.generative_model.parameters(), lr=1e-3)
        self.loss_fn = lambda x_hat, x: F.binary_cross_entropy_with_logits(x_hat.view(x_hat.size(0), -1),
                                                                           x.view(x.size(0), -1), reduction='sum') / \
                                        x.shape[0]

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
            # print(self.transition)
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
        plot_interval = self.plot_interval
        num_ep = self.max_episode
        num_frames = self.max_step
        self.total_step = 0
        """     Log initialization    """
        g_losses = []
        eval_losses = []

        """      Train the agent      """
        for self.episode in range(1, num_ep + 1):
            self.is_test = False

            state = self.env.reset()
            print(f"episode:{self.episode}-{len(self.memory)}")
            for step in range(1, num_frames + 1):
                self.total_step+=1
                """ get channel in terms of state """
                selected_action = self.select_action(state)
                state_next, reward, done, info = self.step()
                state = state_next
                # if training is ready
                if (
                    len(self.memory)>=self.batch_size*50
                ):
                    gloss,eloss = self.update_model(step)
                    g_losses.append(gloss)
                    eval_losses.append(eloss)
                # plotting
                # if self.total_step % plot_interval==0:
                #     self._plot(
                #         self.total_step,
                #         g_losses,
                #         [],
                #         [],
                #     )
                #     pass
        if self.save_flag:
            save_results(
                g_losses,
                eval_losses,
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

