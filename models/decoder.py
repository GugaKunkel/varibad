import torch.nn as nn
from torch.nn import functional as F


class RewardDecoder(nn.Module):
    def __init__(self, layers, latent_dim, num_states):
        super().__init__()
        self.fc_layers = nn.ModuleList()
        in_dim = latent_dim
        for out_dim in layers:
            self.fc_layers.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
        self.fc_out = nn.Linear(in_dim, num_states)

    def forward(self, latent_state):
        h = latent_state
        for layer in self.fc_layers:
            h = F.relu(layer(h))
        return self.fc_out(h)
