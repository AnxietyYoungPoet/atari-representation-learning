import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from a2c_ppo_acktr.utils import init

import os
import numpy as np
from torch.utils.data import RandomSampler, BatchSampler
from .trainer import Trainer
from .utils import EarlyStopping


class Unflatten(nn.Module):
    def __init__(self, new_shape):
        super().__init__()
        self.new_shape = new_shape

    def forward(self, x):
        x_uf = x.view(-1, *self.new_shape)
        return x_uf


class Decoder(nn.Module):
    def __init__(self, feature_size, final_conv_size, final_conv_shape, num_input_channels, encoder_type="Nature"):
        super().__init__()
        self.feature_size = feature_size
        self.final_conv_size = final_conv_size
        self.final_conv_shape = final_conv_shape
        self.num_input_channels = num_input_channels
        # self.fc =
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))
        if encoder_type == "Nature":
            self.main = nn.Sequential(
                nn.Linear(in_features=self.feature_size,
                          out_features=self.final_conv_size),
                nn.ReLU(),
                Unflatten(self.final_conv_shape),

                init_(nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)),
                nn.ReLU(),
                init_(nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=0)),
                nn.ReLU(),
                init_(nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=0,
                                         output_padding=1)),
                nn.ReLU(),
                init_(nn.ConvTranspose2d(in_channels=32, out_channels=num_input_channels,
                                         kernel_size=8, stride=4, output_padding=(2, 0))),
                nn.Sigmoid()
            )

    def forward(self, f):
        im = self.main(f)
        return im


class AE(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.feature_size = self.encoder.feature_size
        self.final_conv_size = self.encoder.final_conv_size
        self.final_conv_shape = self.encoder.final_conv_shape
        self.input_channels = self.encoder.input_channels
        self.logvar_fc = nn.Linear(in_features=self.final_conv_size, out_features=self.feature_size)

        self.decoder = Decoder(feature_size=self.feature_size,
                               final_conv_size=self.final_conv_size,
                               final_conv_shape=self.final_conv_shape,
                               num_input_channels=self.input_channels)

    def forward(self, x):
        final_conv = self.encoder.main[:-1](x)
        mu = self.encoder.main[-1](final_conv)
        logvar = self.logvar_fc(final_conv)
        std = F.softplus(logvar)
        dist = Normal(mu, std)
        if self.training:
            z = dist.rsample()
        else:
            z = mu
        x_hat = self.decoder(z)
        return z, mu, dist, x_hat


class IBAENCETrainer(Trainer):
    # TODO: Make it work for all modes, right now only it defaults to pcl.
    def __init__(self, encoder, config, device=torch.device('cpu'), wandb=None):
        super().__init__(encoder, wandb, device)
        self.config = config
        self.patience = self.config["patience"]
        self.AE = AE(encoder).to(device)
        self.classfier = nn.Linear(self.encoder.hidden_size, self.encoder.hidden_size).to(device)
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.beta = config['beta']
        self.device = device
        self.optimizer = torch.optim.Adam(list(self.AE.parameters()) + list(self.classfier.parameters()),
                                          lr=config['lr'], eps=1e-5)
        self.early_stopper = EarlyStopping(patience=self.patience, verbose=False, wandb=self.wandb, name="encoder")

    def generate_batch(self, episodes):
        total_steps = sum([len(e) for e in episodes])
        print('Total Steps: {}'.format(total_steps))
        # Episode sampler
        # Sample `num_samples` episodes then batchify them with `self.batch_size` episodes per batch
        sampler = BatchSampler(RandomSampler(range(len(episodes)),
                                             replacement=True, num_samples=total_steps),
                               self.batch_size, drop_last=True)
        for indices in sampler:
            episodes_batch = [episodes[x] for x in indices]
            x_t, x_tprev, x_that, ts, thats = [], [], [], [], []
            for episode in episodes_batch:
                # Get one sample from this episode
                t, t_hat = 0, 0
                t, t_hat = np.random.randint(1, len(episode)), np.random.randint(0, len(episode))
                x_t.append(episode[t])
                x_tprev.append(episode[t - 1])
            yield torch.stack(x_t).float().to(self.device) / 255., torch.stack(x_tprev).float().to(self.device) / 255.

    def do_one_epoch(self, epoch, episodes):
        mode = "train" if self.AE.training and self.classfier.training else "val"
        epoch_loss, steps = 0., 0
        epoch_kl_loss, epoch_nce_loss, epoch_recon_loss = 0., 0., 0.
        data_generator = self.generate_batch(episodes)
        for x_t, x_tprev in data_generator:
            with torch.set_grad_enabled(mode == 'train'):
                z_t, mu, dist, x_hat = self.AE(x_t)
                N = z_t.size(0)
                z_tprev, _, _, _ = self.AE(x_tprev)
                prior_dist = Normal(
                    torch.zeros_like(z_t).to(self.device), torch.ones_like(z_t).to(self.device))
                kl_loss = self.beta * kl_divergence(dist, prior_dist).sum(1).mean()
                predictions = self.classfier(z_t)
                logits = torch.matmul(predictions, z_tprev.t())
                nce_loss = F.cross_entropy(logits, torch.arange(N).to(self.device))
                recon_loss = F.mse_loss(x_hat, x_t, reduction='sum') * 0.001
                loss = kl_loss + nce_loss + recon_loss

            if mode == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            epoch_loss += loss.detach().item()
            epoch_kl_loss += kl_loss.detach().item()
            epoch_nce_loss += nce_loss.detach().item()
            epoch_recon_loss += recon_loss.detach().item()
            steps += 1

        self.log_results(
            epoch, epoch_loss / steps, epoch_kl_loss / steps,
            epoch_nce_loss / steps, epoch_recon_loss / steps, prefix=mode)
        if mode == "val":
            self.early_stopper(-epoch_loss / steps, self.encoder)

    #             xim = x_hat.detach().cpu().numpy()[0].transpose(1,2,0)
    #             self.wandb.log({"example_reconstruction": [self.wandb.Image(xim, caption="")]})

    def train(self, tr_eps, val_eps):
        for e in range(self.epochs):
            self.AE.train()
            self.do_one_epoch(e, tr_eps)

            self.AE.eval()
            self.do_one_epoch(e, val_eps)

            if self.early_stopper.early_stop:
                break
        torch.save(self.encoder.state_dict(), os.path.join(self.wandb.run.dir, self.config['env_name'] + '.pt'))

    def log_results(
        self, epoch_idx, epoch_loss, epoch_kl_loss,
        epoch_nce_loss, epoch_recon_loss, prefix=''
    ):
        print(
            f'{prefix.capitalize()} Epoch: {epoch_idx}, Epoch Loss: {epoch_loss}, {prefix.capitalize()}')

        self.wandb.log({
            prefix + '_loss': epoch_loss,
            prefix + '_kl_loss': epoch_kl_loss,
            prefix + '_nce_loss': epoch_nce_loss,
            prefix + '_recon_loss': epoch_recon_loss,
        }, step=epoch_idx, commit=False)
