import torch
import torch.nn.functional as F
import torch.optim as optim

from agents.GAI.LICE import *
from agents.GAI.AE import *
from agents.GAI.VAE import *
from agents.GAI.GAN import *

class agent_utils:
    def __init__(self):
        self.generative_model = LICE().to(self.device)
        self.generative_opt = optim.Adam(self.generative_model.parameters(), lr=1e-3)

    def update_model(self):
        """Update the model by gradient descent."""
        device = self.device  # for shortening the following lines

        samples = self.memory.sample_batch()
        channel_g = torch.FloatTensor(samples["obs"]).to(device)
        true_precoder = torch.FloatTensor(samples["next_obs"]).to(device)

        pred_precoder = self.generative_model(channel_g)
        gai_loss = F.mse_loss(pred_precoder, true_precoder)

        self.generative_opt.zero_grad()
        gai_loss.backward()
        self.generative_opt.step()

        return gai_loss.data