import torch
import torch.nn as nn
from torch.nn import functional as F

from utils import helpers as utl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class StateTransitionDecoder(nn.Module):
    def __init__(self,
                 args,
                 layers,
                 latent_dim,
                 action_dim,
                 action_embed_dim,
                 state_dim,
                 state_embed_dim
                 ):
        super(StateTransitionDecoder, self).__init__()

        self.args = args

        self.state_encoder = utl.FeatureExtractor(state_dim, state_embed_dim, F.relu)
        self.action_encoder = utl.FeatureExtractor(action_dim, action_embed_dim, F.relu)

        curr_input_dim = latent_dim + state_embed_dim + action_embed_dim
        self.fc_layers = nn.ModuleList([])
        for i in range(len(layers)):
            self.fc_layers.append(nn.Linear(curr_input_dim, layers[i]))
            curr_input_dim = layers[i]

        # output layer
        self.fc_out = nn.Linear(curr_input_dim, state_dim)

    def forward(self, latent_state, state, actions):
        ha = self.action_encoder(actions)
        hs = self.state_encoder(state)
        h = torch.cat((latent_state, hs, ha), dim=-1)

        for i in range(len(self.fc_layers)):
            h = F.relu(self.fc_layers[i](h))

        return self.fc_out(h)


class RewardDecoder(nn.Module):
    def __init__(self,
                 args,
                 layers,
                 latent_dim,
                 num_states,
                 input_prev_state=True,
                 input_action=True,
                 ):
        super(RewardDecoder, self).__init__()

        self.args = args

        self.input_prev_state = input_prev_state
        self.input_action = input_action

        # one output head per state to predict rewards
        curr_input_dim = latent_dim
        self.fc_layers = nn.ModuleList([])
        for i in range(len(layers)):
            self.fc_layers.append(nn.Linear(curr_input_dim, layers[i]))
            curr_input_dim = layers[i]
        self.fc_out = nn.Linear(curr_input_dim, num_states)

    def forward(self, latent_state, next_state, prev_state=None, actions=None):
        h = latent_state.clone()
        for i in range(len(self.fc_layers)):
            h = F.relu(self.fc_layers[i](h))
        return self.fc_out(h)
