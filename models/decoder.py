import torch
import torch.nn as nn
from torch.nn import functional as F


class RewardDecoder(nn.Module):
    def __init__(self,
                 layers,
                 latent_dim,
                 num_states,
                 ):
        super(RewardDecoder, self).__init__()
        # one output head per state to predict rewards
        curr_input_dim = latent_dim
        self.fc_layers = nn.ModuleList([])
        for i in range(len(layers)):
            self.fc_layers.append(nn.Linear(curr_input_dim, layers[i]))
            curr_input_dim = layers[i]
        self.fc_out = nn.Linear(curr_input_dim, num_states)

    def forward(self, latent_state):
        h = latent_state.clone()
        for i in range(len(self.fc_layers)):
            h = F.relu(self.fc_layers[i](h))
        return self.fc_out(h)
