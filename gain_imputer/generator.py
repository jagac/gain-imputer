import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, dim: int, h_dim: int) -> None:
        """
        Generator module for the GAIN (Generative Adversarial Imputation Network) model.
        This module is responsible for learning and generating imputed values for missing data
        based on the observed values and hints.

        :param dim: The dimensionality of the input data.
        :param h_dim: The dimensionality of the hidden layer in the generator.
        """
        super().__init__()
        self.w1 = nn.Linear(dim * 2, h_dim)
        self.w2 = nn.Linear(h_dim, h_dim)
        self.w3 = nn.Linear(h_dim, dim)
        self.b1 = nn.Parameter(torch.zeros(h_dim))
        self.b2 = nn.Parameter(torch.zeros(h_dim))
        self.b3 = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([x, m], dim=1)
        h1 = torch.relu(self.w1(inputs) + self.b1)
        h2 = torch.relu(self.w2(h1) + self.b2)
        prob = torch.sigmoid(self.w3(h2) + self.b3)

        return prob
