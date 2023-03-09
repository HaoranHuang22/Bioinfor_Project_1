# Import libraries
import numpy as np
import random
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import math
from tqdm import tqdm
from torch import optim

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original paper
    https://arxiv.org/abs/2006.11239
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    https://openreview.net/forum?id=-NEXDKk8gZ
    """
    t_T = torch.linspace(0, timesteps, timesteps + 1, dtype = torch.float64) / timesteps
    f_t = torch.cos((t_T + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = f_t / f_t[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class Diffusion(nn.Module):
    def __init__(self, timesteps=1000, beta_schedule = 'cosine', device = None, data_size = (1, 20, 3)) -> None:
        super(Diffusion, self).__init__()
        self.device = device
        self.data_size = data_size # batch size, subsample size, 3
        self.timesteps = timesteps
        
        if beta_schedule == "linear":
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == "cosine":
            beta_schedule_fn = cosine_beta_schedule
        else:
            raise ValueError(f'unkown beta schedule {beta_schedule}')
        
        self.betas = beta_schedule_fn(self.timesteps)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim = 0).to(self.device)

    def sample_timesteps(self, batch_size: int):
        return torch.randint(low=1, high=self.timesteps, size=(batch_size, ))
    
    def q_sample(self, x: torch.Tensor, t: torch.Tensor):
        alpha_bars_t = self.alpha_bars[t].view(-1, 1, 1, 1)
        epsilon = torch.rand_like(x, device=self.device)
        
        noisy = torch.sqrt(alpha_bars_t) * x + torch.sqrt(1 - alpha_bars_t) * epsilon
        return noisy, epsilon

    
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int = 128, dropout: float = 0.1, max_len: int = 1000, apply_dropout: bool = False):
        """
        sinusoial embedding for time steps
        Use Attention is all you need settings

        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.apply_dropout = apply_dropout

        pos_encoding = torch.zeros(max_len, embedding_dim)
        position = torch.arange(start=0, end=max_len).unsqueeze(1) #dim: [max_len, 1]
        div_term = torch.exp(-math.log(10000.0) * torch.arange(0, embedding_dim, 2).float() / embedding_dim)

        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer(name='pos_encoding', tensor=pos_encoding, persistent=False) # not included in the module's state_dict
    def forward(self, t):

        positional_encoding = self.pos_encoding[t].squeeze(1)
        if self.apply_dropout:
            return self.dropout(positional_encoding)
        return positional_encoding


    
class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels:int, residual: bool = False):
        """
        All models have two convolutional residual blocks mentioned by original DDPM paper
        """
        super(DoubleConv, self).__init__()
        self.residual = residual
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size=(3,3), padding=(1,1), bias=False),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
            nn.GELU(),
            nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size=(3,3), padding=(1,1), bias=False),
            nn.GroupNorm(num_groups=1, num_channels=out_channels)
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int = 128):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2,2)),
            DoubleConv(in_channels=in_channels, out_channels=in_channels, residual=True),
            DoubleConv(in_channels=in_channels, out_channels=out_channels),
        )
        
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=emb_dim, out_features=out_channels)
        )
    def forward(self, x: torch.Tensor, t_embedding: torch.Tensor):
        """
        Downsamples input tensor, calculates embedding and adds embedding channel wise

        If, `x.shape == [128, 1, 20, 3]` and `out_channels = 64`, then max_conv outputs [128, 64, 10, 1]

        `t_embedding` is embedding of timestep of shape [batch, time_dim]

        """
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t_embedding)
        emb = emb.view(emb.shape[0], emb.shape[1], 1, 1).repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb
    
class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int = 128):
        super(Up, self).__init__()
        self.up = nn.Upsample(scale_factor=(2,3), mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels=in_channels, out_channels=in_channels, residual=True),
            DoubleConv(in_channels=in_channels, out_channels=out_channels)
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=emb_dim, out_features=out_channels)
        )
    
    def forward(self, x:torch.Tensor, x_skip: torch.Tensor, t_embedding: torch.Tensor):
        x = self.up(x)
        
        x = torch.cat([x_skip, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t_embedding)
        emb = emb.view(emb.shape[0], emb.shape[1], 1, 1).repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb
    
class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, noise_steps: int = 1000, time_dim: int = 128):
        super(UNet, self).__init__()

        self.time_dim = time_dim
        self.pos_encoding = PositionalEncoding(embedding_dim=time_dim, max_len=noise_steps)
        
        self.input_conv = DoubleConv(in_channels, 32)
        self.down = Down(32, 64)

        self.bottleneck1 = DoubleConv(64, 128)
        self.bottleneck2 = DoubleConv(128, 128)
        self.bottleneck3 = DoubleConv(128, 64)

        self.up = Up(96, 32)
        self.out_conv = nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=(1,1))

    def forward(self, x: torch.Tensor, t: torch.LongTensor):
        """
        Forward pass with coordinate tensor and timestep 
        """

        t = self.pos_encoding(t)
        x1 = self.input_conv(x)
        x2 = self.down(x1, t)

        x3 = self.bottleneck1(x2)
        x3 = self.bottleneck2(x3)
        x3 = self.bottleneck3(x3)

        x = self.up(x3, x1, t)
        x = self.out_conv(x)
        
        return x
    



