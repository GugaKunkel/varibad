"""
Based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr
Used for on-policy rollout storages.
"""
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from utils import helpers as utl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class OnlineStorage(object):
    def __init__(self, args, num_steps, num_processes, state_dim, belief_dim, hidden_size):
        self.args = args
        self.num_steps = num_steps  # how many steps to do per update (= size of online buffer)
        self.num_processes = num_processes  # number of parallel processes
        self.step = 0  # keep track of current environment step
        
        # inputs to the policy. This will include s_0 when state was reset (hence num_steps+1)
        self.prev_state = torch.zeros(num_steps + 1, num_processes, state_dim)
        
        # latent variables (of VAE)
        self.latent_samples = []
        self.latent_mean = []
        self.latent_logvar = []
        
        # hidden states of RNN (necessary if we want to re-compute embeddings)
        self.hidden_states = torch.zeros(num_steps + 1, num_processes, hidden_size)
        if self.args.pass_belief_to_policy:
            self.beliefs = torch.zeros(num_steps + 1, num_processes, belief_dim)
        else:
            self.beliefs = None
        
        # rewards and end of episodes
        self.rewards_raw = torch.zeros(num_steps, num_processes, 1)
        self.rewards_normalised = torch.zeros(num_steps, num_processes, 1)
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        
        # actions
        self.actions = torch.zeros(num_steps, num_processes, 1, dtype=torch.long)
        self.action_log_probs = None
        
        # values and returns
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.to_device()
    
    def to_device(self):
        self.prev_state = self.prev_state.to(device)
        self.latent_samples = [t.to(device) for t in self.latent_samples]
        self.latent_mean = [t.to(device) for t in self.latent_mean]
        self.latent_logvar = [t.to(device) for t in self.latent_logvar]
        self.hidden_states = self.hidden_states.to(device)
        if self.args.pass_belief_to_policy:
            self.beliefs = self.beliefs.to(device)
        self.rewards_raw = self.rewards_raw.to(device)
        self.rewards_normalised = self.rewards_normalised.to(device)
        self.masks = self.masks.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.actions = self.actions.to(device)
    
    def insert(self,
                state,
                belief,
                actions,
                rewards_raw,
                rewards_normalised,
                value_preds,
                masks,
                hidden_states=None,
                latent_sample=None,
                latent_mean=None,
                latent_logvar=None,
                ):
        self.prev_state[self.step + 1].copy_(state)
        if self.args.pass_belief_to_policy:
            self.beliefs[self.step + 1].copy_(belief)
        self.latent_samples.append(latent_sample.detach().clone())
        self.latent_mean.append(latent_mean.detach().clone())
        self.latent_logvar.append(latent_logvar.detach().clone())
        self.hidden_states[self.step + 1].copy_(hidden_states.detach())
        self.actions[self.step] = actions.detach().clone()
        self.rewards_raw[self.step].copy_(rewards_raw)
        self.rewards_normalised[self.step].copy_(rewards_normalised)
        if isinstance(value_preds, list):
            self.value_preds[self.step].copy_(value_preds[0].detach())
        else:
            self.value_preds[self.step].copy_(value_preds.detach())
        self.masks[self.step + 1].copy_(masks)
        self.step = (self.step + 1) % self.num_steps
    
    def after_update(self):
        self.prev_state[0].copy_(self.prev_state[-1])
        if self.args.pass_belief_to_policy:
            self.beliefs[0].copy_(self.beliefs[-1])
        self.latent_samples = []
        self.latent_mean = []
        self.latent_logvar = []
        self.hidden_states[0].copy_(self.hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])
        self.action_log_probs = None
    
    def compute_returns(self, next_value, gamma, tau):
        rewards = self.rewards_normalised.clone()
        self.value_preds[-1] = next_value
        gae = 0
        for step in reversed(range(rewards.size(0))):
            delta = rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
            gae = delta + gamma * tau * self.masks[step + 1] * gae
            self.returns[step] = gae + self.value_preds[step]
    
    def before_update(self, policy):
        latent = utl.get_latent_for_policy(latent_sample=torch.stack(self.latent_samples[:-1]),
                                            latent_mean=torch.stack(self.latent_mean[:-1]),
                                            latent_logvar=torch.stack(self.latent_logvar[:-1]))
        _, action_log_probs, _ = policy.evaluate_actions(self.prev_state[:-1],
                                                        latent,
                                                        self.beliefs[:-1] if self.beliefs is not None else None,
                                                        self.actions)
        self.action_log_probs = action_log_probs.detach()
    
    def feed_forward_generator(self, advantages, num_mini_batch):
        num_steps, num_processes = self.rewards_raw.size()[0:2]
        batch_size = num_processes * num_steps
        assert batch_size >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "* number of steps ({}) = {} "
            "to be greater than or equal to the number of PPO mini batches ({})."
            "".format(num_processes, num_steps, num_processes * num_steps,
                        num_mini_batch))
        mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True)
        for indices in sampler:
            state_batch = self.prev_state[:-1].reshape(-1, *self.prev_state.size()[2:])[indices]
            latent_sample_batch = torch.cat(self.latent_samples[:-1])[indices]
            latent_mean_batch = torch.cat(self.latent_mean[:-1])[indices]
            latent_logvar_batch = torch.cat(self.latent_logvar[:-1])[indices]
            if self.args.pass_belief_to_policy:
                belief_batch = self.beliefs[:-1].reshape(-1, *self.beliefs.size()[2:])[indices]
            else:
                belief_batch = None
            
            actions_batch = self.actions.reshape(-1, self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].reshape(-1, 1)[indices]
            return_batch = self.returns[:-1].reshape(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.reshape(-1, 1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.reshape(-1, 1)[indices]
            
            yield state_batch, belief_batch, actions_batch, \
                    latent_sample_batch, latent_mean_batch, latent_logvar_batch, \
                    value_preds_batch, return_batch, old_action_log_probs_batch, adv_targ
