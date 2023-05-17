import torch
import torch.nn.functional as F
import torch.optim as optim

from agents.GAI.LICE import *
from agents.GAI.AE import *
from agents.GAI.VAE import *
from agents.GAI.GAN import *

class agent_utils:
    def __init__(self, args):
        pass

    def update_model(self, t):
        """Update the model by gradient descent."""
        samples = self.memory.sample_batch()

        channel_g = torch.FloatTensor(samples["obs"]).to(self.device)
        true_precoder = torch.FloatTensor(samples["obs"]).to(self.device)

        gloss = self.generative_model.train_step(channel_g, t, self.loss_fn)
        # print(channel_g)
        gai_loss = gloss['loss']
        eval_loss = gloss['e-loss']
        print(eval_loss)
        self.generative_opt.zero_grad()
        gai_loss.backward()
        self.generative_opt.step()
        return gai_loss.data, eval_loss.data