import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, dim: int, h_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim * 2, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim // 2)
        self.fc3 = nn.Linear(h_dim // 2, dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def xavier_init(self):
        layers = [self.fc1, self.fc2, self.fc3]
        [torch.nn.init.xavier_normal_(layer.weight) for layer in layers]

    def forward(
        self, x: torch.Tensor, z: torch.Tensor, m: torch.Tensor
    ) -> torch.Tensor:
        inp = m * x + (1 - m) * z
        inp = torch.cat((inp, m), dim=1)

        out = self.relu(self.fc1(inp))
        out = self.relu(self.fc2(out))
        out = self.sigmoid(self.fc3(out))

        return out
