import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_neurons, hidden_neurons, out_neurons, kernel_size):
        super().__init__()
        self.mlp1 = nn.Conv3d(in_neurons, hidden_neurons, kernel_size)
        self.mlp2 = nn.Conv3d(hidden_neurons, out_neurons, kernel_size)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x