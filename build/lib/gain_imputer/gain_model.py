from typing import Tuple

import torch
import torch.nn as nn

from gain_imputer.discriminator import Discriminator
from gain_imputer.generator import Generator


class Gain(nn.Module):
    def __init__(self, dim: int, h_dim: int) -> None:
        super().__init__()
        self.generator = Generator(dim, h_dim)
        self.discriminator = Discriminator(dim, h_dim)

    def forward(
        self, x: torch.Tensor, m: torch.Tensor, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        g_sample = self.generator(x, m)
        hat_x = x * m + g_sample * (1 - m)
        d_prob = self.discriminator(hat_x, h)

        return g_sample, hat_x, d_prob
