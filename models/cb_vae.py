"""
Based on: https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
"""
import torch
import torch.nn as nn
from torch.nn import functional as F


def apply_init_(modules):
    """
    Initialize NN modules
    """
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            m.bias.data.zero_()


class CBVAE(nn.Module):
    def __init__(self,
                 obs_shape,
                 latent_dim=1,
                 channels=[32, 64, 128, 256, 512]):
        super(CBVAE, self).__init__()

        # build encoder
        in_channels = obs_shape[0]
        modules = list()
        for ch in channels:
            modules.append(nn.Sequential(nn.Conv2d(in_channels,
                                                   out_channels=ch,
                                                   kernel_size=3,
                                                   stride=2,
                                                   padding=1),
                                         nn.ReLU(inplace=True)))
            in_channels = ch
        self.encoder = nn.Sequential(*modules)

        self.fc_mu = nn.Linear(channels[-1]*4, latent_dim)
        self.fc_log_var = nn.Linear(channels[-1]*4, latent_dim)

        # build decoder
        self.decoder_input = nn.Linear(latent_dim, channels[-1]*4)

        channels.reverse()
        modules = list()
        for i in range(len(channels)-1):
            modules.append(nn.Sequential(nn.ConvTranspose2d(channels[i],
                                                            channels[i+1],
                                                            kernel_size=3,
                                                            stride=2,
                                                            padding=1,
                                                            output_padding=1),
                                         nn.ReLU(inplace=True)))
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(nn.ConvTranspose2d(channels[-1],
                                                            channels[-1],
                                                            kernel_size=3,
                                                            stride=2,
                                                            padding=1,
                                                            output_padding=1),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(channels[-1],
                                                   out_channels=3,
                                                   kernel_size=3,
                                                   padding=1))

        # initialise all layers
        apply_init_(self.modules())

        # put model into train mode
        self.train()

    def reparameterise(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + std*eps

    def encode(self, x):
        """
        Parameters:
        -----------
        x : `torch.Tensor`
            [N x C x H x W]

        Returns:
        --------
        mu : `torch.Tensor`
            [N x latent_dim]
        log_var : `torch.Tensor`
            [N x latent_dim]
        """
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return self.reparameterise(mu, log_var), mu, log_var

    def decode(self, z):
        """
        Parameters:
        -----------
        z : `torch.Tensor`
            [N x latent_dim]

        Returns:
        --------
        x : `torch.Tensor`
            [N x C x H x W]
        """
        x = self.decoder_input(z)
        x = x.view(-1, 512, 2, 2)
        x = self.decoder(x)
        return torch.sigmoid(self.final_layer(x))

    def forward(self, x):
        z, mu, log_var = self.encode(x)
        return self.decode(z), mu, log_var


def sumlogC(x, eps=1e-5):
    """
    Numerically stable implementation of sum of logarithm of Continuous Bernoulli constant C
    Returns log normalising constant for x in (0, x-eps) and (x+eps, 1)
    Uses Taylor 3rd degree approximation in [x-eps, x+eps].

    Parameter
    ----------
    x : `torch.Tensor`
        [batch_size x C x H x W]. x takes values in (0,1).
    """
    # clip x such that x in (0, 1)
    x = torch.clamp(x, eps, 1.-eps)
    # get mask if x is not in [0.5-eps, 0.5+eps]
    mask = torch.abs(x - .5).ge(eps)
    # points that are (0, 0.5-eps) and (0.5+eps, 1)
    far = x[mask]
    # points that are [0.5-eps, 0.5+eps]
    close = x[~mask]
    # Given by log(|2tanh^-1(1-2x)|) - log(|1-2x|)
    far_values = torch.log(torch.abs(2.*torch.atanh(1-2.*far))) - torch.log(torch.abs(1-2.*far))
    # using Taylor expansion to 3rd degree
    close_values = torch.log(2. + (1-2*close).pow(2)/3 + (1-2*close).pow(4)/5)
    return far_values.sum() + close_values.sum()


def cb_vae_loss(recon_x, x, mu, log_var):
    """
    recon_x : [batch_size x C x H x W]
    x : [batch_size x C x H x W]
    mu : [batch_size, z_dim)]
    log_var: [batch_size, z_dim]
    """
    BCE  = F.binary_cross_entropy(recon_x, x, reduction='sum')
    logC = sumlogC(recon_x)
    KL   = 0.5 * torch.sum(mu.pow(2) + log_var.exp() - log_var - 1)
    return (KL + BCE - logC) / x.size(0)
