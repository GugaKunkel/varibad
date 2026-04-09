"""
Based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr
"""
import torch
import torch.nn as nn
from utils import helpers as utl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Policy(nn.Module):
    def __init__(self,
                args,
                # input
                pass_belief_to_policy,
                dim_state,
                dim_latent,
                dim_belief,
                hidden_layers,
                # output
                action_space
                ):
        """
        The policy can get any of these as input:
        - state (given by environment)
        - latent variable (from VAE)
        """
        super(Policy, self).__init__()
        self.args = args
        self.activation_function = nn.Tanh()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('tanh'))

        self.pass_belief_to_policy = pass_belief_to_policy

        # set normalisation parameters for the inputs (will be updated from outside using the RL batches)
        self.state_rms = utl.RunningMeanStd(shape=(dim_state))
        self.latent_rms = utl.RunningMeanStd(shape=(dim_latent))
        if self.pass_belief_to_policy:
            self.belief_rms = utl.RunningMeanStd(shape=(dim_belief))

        curr_input_dim = dim_state + dim_latent + dim_belief * int(self.pass_belief_to_policy)
        self.state_encoder = utl.FeatureExtractor(dim_state, self.args.policy_state_embedding_dim, self.activation_function)
        curr_input_dim = curr_input_dim - dim_state + self.args.policy_state_embedding_dim
        self.latent_encoder = utl.FeatureExtractor(dim_latent, self.args.policy_latent_embedding_dim, self.activation_function)
        curr_input_dim = curr_input_dim - dim_latent + self.args.policy_latent_embedding_dim
        self.use_belief_encoder = self.args.policy_belief_embedding_dim is not None
        if self.pass_belief_to_policy and self.use_belief_encoder:
            self.belief_encoder = utl.FeatureExtractor(dim_belief, self.args.policy_belief_embedding_dim, self.activation_function)
            curr_input_dim = curr_input_dim - dim_belief + self.args.policy_belief_embedding_dim

        # initialise actor and critic
        hidden_layers = [int(h) for h in hidden_layers]
        self.actor_layers = nn.ModuleList()
        self.critic_layers = nn.ModuleList()
        in_dim = curr_input_dim
        for out_dim in hidden_layers:
            self.actor_layers.append(init_(nn.Linear(in_dim, out_dim)))
            self.critic_layers.append(init_(nn.Linear(in_dim, out_dim)))
            in_dim = out_dim
        self.critic_linear = nn.Linear(in_dim, 1)

        # output distributions of the policy
        num_outputs = action_space.n
        self.dist = Categorical(hidden_layers[-1], num_outputs)
    
    def forward_actor(self, inputs):
        h = inputs
        for layer in self.actor_layers:
            h = self.activation_function(layer(h))
        return h
    
    def forward_critic(self, inputs):
        h = inputs
        for layer in self.critic_layers:
            h = self.activation_function(layer(h))
        return h
    
    def forward(self, state, latent, belief):
        # handle inputs (normalise + embed)
        state = (state - self.state_rms.mean) / torch.sqrt(self.state_rms.var + 1e-8)
        state = self.state_encoder(state)
        latent = (latent - self.latent_rms.mean) / torch.sqrt(self.latent_rms.var + 1e-8)
        latent = self.latent_encoder(latent)
        if self.pass_belief_to_policy:
            belief = (belief - self.belief_rms.mean) / torch.sqrt(self.belief_rms.var + 1e-8)
            if self.use_belief_encoder:
                belief = self.belief_encoder(belief.float())
        else:
            belief = torch.zeros(0, ).to(device)
        # concatenate inputs
        inputs = torch.cat((state, latent, belief), dim=-1)

        # forward through critic/actor part
        hidden_critic = self.forward_critic(inputs)
        hidden_actor = self.forward_actor(inputs)
        return self.critic_linear(hidden_critic), hidden_actor

    def act(self, state, latent, belief, deterministic=False):
        """ Returns the (raw) actions and their value. """
        value, actor_features = self.forward(state=state, latent=latent, belief=belief)
        dist = self.dist(actor_features)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        return value, action
    
    def get_value(self, state, latent, belief):
        value, _ = self.forward(state, latent, belief)
        return value
    
    def update_rms(self, policy_storage):
        """ Update normalisation parameters for inputs with current data """
        self.state_rms.update(policy_storage.prev_state[:-1])
        latent = utl.get_latent_for_policy(torch.cat(policy_storage.latent_samples[:-1]),
                                            torch.cat(policy_storage.latent_mean[:-1]),
                                            torch.cat(policy_storage.latent_logvar[:-1])
                                            )
        self.latent_rms.update(latent)
        if self.pass_belief_to_policy:
            self.belief_rms.update(policy_storage.beliefs[:-1])
    
    def evaluate_actions(self, state, latent, belief, action):
        value, actor_features = self.forward(state, latent, belief)
        dist = self.dist(actor_features)
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()
        return value, action_log_probs, dist_entropy

def init(module, weight_init, bias_init, gain=1.0):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return super().log_prob(actions.squeeze(-1)).unsqueeze(-1)

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)

class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01)
        self.linear = init_(nn.Linear(num_inputs, num_outputs))
    
    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)
