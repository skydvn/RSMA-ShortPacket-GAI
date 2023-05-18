import torch
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from agents.GAI.models.losses import *


class NaiveEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(NaiveEncoder, self).__init__()
        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

        self.training = True

    def forward(self, x):
        h_ = self.LeakyReLU(self.FC_input(x))
        h_ = self.LeakyReLU(self.FC_input2(h_))
        h_ = self.LeakyReLU(self.FC_input2(h_))
        h_ = self.LeakyReLU(self.FC_input2(h_))
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)  # encoder produces mean and log of variance
        #             (i.e., parateters of simple tractable normal distribution "q"
        outs = torch.cat((mean,log_var), dim=-1)
        return outs


class NaiveDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(NaiveDecoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        h = torch.sigmoid(self.FC_hidden(x))
        h = torch.sigmoid(self.FC_hidden2(h))
        h = self.LeakyReLU(self.FC_hidden2(h))
        h = self.LeakyReLU(self.FC_hidden2(h))

        x_hat = self.FC_output(h)
        return x_hat


class VAE(Module):
    def __init__(self, encoder, decoder, beta, max_capacity=None, capacity_leadin=None, anneal=1.):
        """ Base VAE class for other models

        Args:
            encoder (nn.Module): Encoder network, outputs size [bs, 2*latents]
            decoder (nn.Module): Decoder network, outputs size [bs, nc, N, N]
            beta (float): Beta value for KL divergence weight
            max_capacity (float): Max capacity for capactiy annealing
            capacity_leadin (int): Capacity leadin, linearly scale capacity up to max over leadin steps
            anneal (float): Annealing rate for KL weighting
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.capacity = max_capacity
        self.capacity_leadin = capacity_leadin
        self.global_step = 0
        self.anneal = anneal

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def load_vae_state(self, state):
        def key_map(s, enc_dec_str):
            idx = s.find(enc_dec_str)
            return s[(idx+len(enc_dec_str)+1):]

        encoder_state = {key_map(k, 'encoder'): v for k, v in state.items() if 'encoder' in k}
        decoder_state = {key_map(k, 'decoder'): v for k, v in state.items() if 'decoder' in k}
        self.encoder.load_state_dict(encoder_state)
        self.decoder.load_state_dict(decoder_state)

    def forward(self, x):
        mu, lv = self.unwrap(self.encode(x))
        return self.decode(self.reparametrise(mu, lv))

    def rep_fn(self, batch):
        x, y = batch
        mu, lv = self.unwrap(self.encode(x))
        return mu

    def main_step(self, batch, batch_nb, loss_fn):

        x = batch
        y = 0
        mu, lv = self.unwrap(self.encode(x))
        z = self.reparameterize(mu, lv)
        x_hat = self.decode(z)
        loss = loss_fn(x_hat, x)
        total_kl = self.compute_kl(mu, lv, mean=False)
        beta_kl = self.control_capacity(total_kl, self.global_step, self.anneal)
        state = self.make_state(batch_nb, x_hat, x, y, mu, lv, z)
        self.global_step += 1
        e_loss = F.mse_loss(x_hat,x)
        tensorboard_logs = {'metric/loss': loss+beta_kl, 'metric/recon_loss': loss, 'metric/total_kl': total_kl,
                            'metric/beta_kl': beta_kl}
        return {'loss': e_loss+beta_kl, 'e-loss': e_loss, 'out': tensorboard_logs, 'state': state}

    def compute_kl(self, mu, lv, mean=False):
        total_kl, dimension_wise_kld, mean_kld = gaussian_kls(mu, lv, mean)
        return total_kl

    def make_state(self, batch_nb, x_hat, x, y, mu, lv, z):
        if batch_nb == 0:
            recon = x_hat[:8]
        else:
            recon = None
        state = {'x': x, 'y': y, 'x_hat': x_hat, 'recon': recon, 'mu': mu, 'lv': lv, 'x1': x, 'x2': y, 'z': z}
        return state

    def control_capacity(self, total_kl, global_step, anneal=1.):
        if self.capacity is not None:
            leadin = 1e5 if self.capacity_leadin is None else self.capacity_leadin
            delta = torch.tensor((self.capacity / leadin) * global_step).clamp(max=self.capacity)
            return (total_kl - delta).abs().clamp(min=0) * self.beta * (anneal ** global_step)
        else:
            return total_kl*self.beta

    def train_step(self, batch, batch_nb, loss_fn):
        return self.main_step(batch, batch_nb, loss_fn)

    def val_step(self, batch, batch_nb, loss_fn):
        return self.main_step(batch, batch_nb, loss_fn)

    def unwrap(self, x):
        return torch.split(x, x.shape[1]//2, dim=1)

    def reparameterize(self, mu, lv):
        if self.training:
            std = torch.exp(0.5 * lv)
            eps = torch.randn_like(std)
            return mu + std * eps
        else:
            return mu