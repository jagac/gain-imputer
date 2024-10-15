from typing import Tuple

import torch
import torch.nn as nn

from .discriminator import Discriminator
from .generator import Generator


class Gain(nn.Module):
    def __init__(self, dim: int, h_dim: int):
        super().__init__()
        self.generator = Generator(dim, h_dim)
        self.generator.xavier_init()
        self.discriminator = Discriminator(dim, h_dim)
        self.discriminator.xavier_init()

    def forward(
        self, x: torch.Tensor, z: torch.Tensor, m: torch.Tensor, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        g_sample = self.generator(x, z, m)
        hat_x = m * x + (1 - m) * g_sample
        d_prob = self.discriminator(x, m, g_sample, h)

        return g_sample, hat_x, d_prob
