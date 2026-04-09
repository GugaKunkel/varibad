import os
import pickle
import random
from distutils.util import strtobool

import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def reset_env(env, args, indices=None, state=None):
    """ env can be many environments or just one """
    # reset all environments
    if (indices is None) or (len(indices) == args.num_processes):
        state = env.reset().float().to(device)
    # reset only the ones given by indices
    else:
        assert state is not None
        for i in indices:
            state[i] = env.reset(index=i)
    
    belief = torch.from_numpy(env.get_belief()).float().to(device) if args.pass_belief_to_policy else None
    return state, belief

def env_step(env, action, args):
    next_obs, reward, terminated, truncated, infos = env.step(action)
    if isinstance(next_obs, list):
        next_obs = [o.to(device) for o in next_obs]
    else:
        next_obs = next_obs.to(device)
    if isinstance(reward, list):
        reward = [r.to(device) for r in reward]
    else:
        reward = reward.to(device)
    
    belief = torch.from_numpy(env.get_belief()).float().to(device) if args.pass_belief_to_policy else None
    return [next_obs, belief], reward, terminated, truncated, infos


def select_action(policy, deterministic, state=None, belief=None, latent_sample=None, latent_mean=None, latent_logvar=None):
    """ Select action using the policy. """
    latent = get_latent_for_policy(latent_sample=latent_sample, latent_mean=latent_mean, latent_logvar=latent_logvar)
    action = policy.act(state=state, latent=latent, belief=belief, deterministic=deterministic)
    if isinstance(action, list) or isinstance(action, tuple):
        value, action = action
    else:
        value = None
    action = action.to(device)
    return value, action


def get_latent_for_policy(latent_sample=None, latent_mean=None, latent_logvar=None):
    if (latent_sample is None) and (latent_mean is None) and (latent_logvar is None):
        return None
    
    latent = torch.cat((latent_mean, latent_logvar), dim=-1)
    
    if latent.shape[0] == 1:
        latent = latent.squeeze(0)
    
    return latent


def update_encoding(encoder, next_obs, action, reward, done, hidden_state):
    # reset hidden state of the recurrent net when we reset the task
    if done is not None:
        hidden_state = encoder.reset_hidden(hidden_state, done)
    
    with torch.no_grad():
        latent_sample, latent_mean, latent_logvar, hidden_state = encoder(actions=action.float(),
                                                                            states=next_obs,
                                                                            rewards=reward,
                                                                            hidden_state=hidden_state,
                                                                            return_prior=False)
    return latent_sample, latent_mean, latent_logvar, hidden_state


class FeatureExtractor(nn.Module):
    """ Used for extrating features for states/actions/rewards """
    def __init__(self, input_size, output_size, activation_function):
        super(FeatureExtractor, self).__init__()
        self.output_size = output_size
        self.activation_function = activation_function
        if self.output_size != 0:
            self.fc = nn.Linear(input_size, output_size)
        else:
            self.fc = None
    
    def forward(self, inputs):
        if self.output_size != 0:
            return self.activation_function(self.fc(inputs))
        else:
            return torch.zeros(0, ).to(device)


def save_obj(obj, folder, name):
    filename = os.path.join(folder, name + '.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    # PyTorch version.
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = torch.zeros(shape).float().to(device)
        self.var = torch.ones(shape).float().to(device)
        self.count = epsilon
    
    def update(self, x):
        x = x.view((-1, x.shape[-1]))
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)
    
    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(self.mean, self.var, self.count, batch_mean, batch_var, batch_count)


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count
    
    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + torch.pow(delta, 2) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count
    
    return new_mean, new_var, new_count


def boolean_argument(value):
    """Convert a string value to boolean."""
    return bool(strtobool(value))

def seed(seed):
    print('Seeding random, torch, numpy.')
    random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)
