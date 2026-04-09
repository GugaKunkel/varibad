import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import helpers as utl


class PPO:
    def __init__(self, args, actor_critic, value_loss_coef, entropy_coef, lr=None, clip_param=0.2, ppo_epoch=5, num_mini_batch=5, eps=None):
        self.args = args
        self.actor_critic = actor_critic # the model
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.optimiser = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
    
    def update(self, policy_storage, compute_vae_loss):
        # -- get action values --
        advantages = policy_storage.returns[:-1] - policy_storage.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        
        # update the normalisation parameters of policy inputs before updating
        self.actor_critic.update_rms(policy_storage=policy_storage)
        
        # call this to make sure that the action_log_probs are computed
        # (needs to be done right here because of some caching thing when normalising actions)
        policy_storage.before_update(self.actor_critic)
        
        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        loss_epoch = 0
        for e in range(self.ppo_epoch):
            data_generator = policy_storage.feed_forward_generator(advantages, self.num_mini_batch)
            for sample in data_generator:
                state_batch, belief_batch, \
                actions_batch, latent_sample_batch, latent_mean_batch, latent_logvar_batch, value_preds_batch, \
                return_batch, old_action_log_probs_batch, adv_targ = sample
                
                state_batch = state_batch.detach()
                latent_sample_batch = latent_sample_batch.detach()
                latent_mean_batch = latent_mean_batch.detach()
                latent_logvar_batch = latent_logvar_batch.detach()
                
                latent_batch = utl.get_latent_for_policy(latent_sample=latent_sample_batch, latent_mean=latent_mean_batch, latent_logvar=latent_logvar_batch)
                
                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy = self.actor_critic.evaluate_actions(state=state_batch, latent=latent_batch, belief=belief_batch, action=actions_batch)
                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()
                
                value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                # use clipped Huber value loss.
                value_losses = F.smooth_l1_loss(values, return_batch, reduction='none')
                value_losses_clipped = F.smooth_l1_loss(value_pred_clipped, return_batch, reduction='none')
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                
                self.optimiser.zero_grad()
                loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.args.policy_max_grad_norm)
                self.optimiser.step()
                
                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                loss_epoch += loss.item()
        
        for _ in range(self.args.num_vae_updates):
            compute_vae_loss(update=True)
        
        num_updates = self.ppo_epoch * self.num_mini_batch
        
        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        loss_epoch /= num_updates
        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, loss_epoch
    
    def act(self, state, latent, belief, deterministic=False):
        return self.actor_critic.act(state=state, latent=latent, belief=belief, deterministic=deterministic)
