import torch
import torch.nn.functional as F
import torch.optim as optim

from agents.GAI.LICE import *
from agents.GAI.AE import *
from agents.GAI.VAE import *
from agents.GAI.GAN import *

class agent_utils:
    def __init__(self, args):
        self.generative_model = VAE(
            NaiveEncoder(input_dim=args.user_num*args.antenna_num,
                         hidden_dim=args.user_num*args.antenna_num/2,
                         latent_dim=args.user_num*args.antenna_num/4),
            NaiveDecoder(latent_dim=args.user_num*args.antenna_num/4,
                         hidden_dim=args.user_num*args.antenna_num/2,
                         output_dim=args.user_num*args.antenna_num),
            args.beta, args.capacity, args.capacity_leadin
        ).to(self.device)
        self.generative_opt = optim.Adam(self.generative_model.parameters(), lr=1e-3)

    def update_model(self):
        """Update the model by gradient descent."""
        samples = self.memory.sample_batch()
        channel_g = torch.FloatTensor(samples["obs"]).to(self.device)
        true_precoder = torch.FloatTensor(samples["obs"]).to(self.device)

        gai_loss = self.generative_model.train_step()

        self.generative_opt.zero_grad()
        gai_loss.backward()
        self.generative_opt.step()

        return gai_loss.data