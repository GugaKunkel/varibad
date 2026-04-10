import os
import time

import numpy as np
import torch

from algorithms.online_storage import OnlineStorage
from algorithms.ppo import PPO
from environments.parallel_envs import make_vec_envs
from models.policy import Policy
from utils import evaluation as utl_eval
from utils import helpers as utl
from utils.tb_logger import TBLogger
from vae import VaribadVAE

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MetaLearner:
    def __init__(self, args):
        self.args = args
        utl.seed(self.args.seed)
        self.logger = TBLogger(self.args, self.args.exp_label)
        
        # calculate number of training updates to do and keep count of frames/iterations
        self.num_updates = int(args.num_frames) // args.policy_num_steps // args.num_processes
        self.frames = 0
        self.iter_idx = -1
        
        # initialise vectorized environments (for parallel training)
        self.envs = make_vec_envs(env_name=args.env_name, seed=args.seed, num_processes=args.num_processes,
                                    gamma=args.policy_gamma, device=device,
                                    episodes_per_task=self.args.max_rollouts_per_task,
                                    normalise_rew=True, ret_rms=None,
                                    tasks=None
                                )
        
        # calculate what the maximum length of the trajectories is
        self.args.max_trajectory_len = self.envs._max_episode_steps * self.args.max_rollouts_per_task
        
        # get policy input dimensions
        self.args.state_dim = self.envs.observation_space.shape[0]
        self.args.belief_dim = self.envs.belief_dim
        self.args.num_states = self.envs.num_states
        
        # get policy output (action) dimensions
        self.args.action_space = self.envs.action_space
        self.args.action_dim = 1
        
        # initialise VAE and policy
        self.vae = VaribadVAE(self.args, lambda: self.iter_idx)
        self.policy_storage = OnlineStorage(args=self.args,
                                            num_steps=self.args.policy_num_steps,
                                            num_processes=self.args.num_processes,
                                            state_dim=self.args.state_dim,
                                            belief_dim=self.args.belief_dim,
                                            hidden_size=self.args.encoder_gru_hidden_size,
                                            )
        self.policy = self.initialise_policy()
    
    def initialise_policy(self):
        policy_net = Policy(
            args=self.args,
            # input
            pass_belief_to_policy=self.args.pass_belief_to_policy,
            dim_state=self.args.state_dim,
            dim_latent=self.args.latent_dim * 2,
            dim_belief=self.args.belief_dim,
            hidden_layers=self.args.policy_layers,
            # output
            action_space=self.envs.action_space
        ).to(device)
        policy = PPO(
            self.args,
            policy_net,
            self.args.policy_value_loss_coef,
            self.args.policy_entropy_coef,
            lr=self.args.lr_policy,
            eps=self.args.policy_eps,
            ppo_epoch=self.args.ppo_num_epochs,
            num_mini_batch=self.args.ppo_num_minibatch,
            clip_param=self.args.ppo_clip_param,
        )
        return policy
    
    def train(self):
        """ Main Meta-Training loop """
        start_time = time.time()
        
        # reset environments
        prev_state, belief = utl.reset_env(self.envs, self.args)
        
        # insert initial observation / embeddings to rollout storage
        self.policy_storage.prev_state[0].copy_(prev_state)
        
        # log once before training
        with torch.no_grad():
            self.log(None, None, start_time)
        
        for self.iter_idx in range(self.num_updates):
            # First, re-compute the hidden states given the current rollouts (since the VAE might've changed)
            with torch.no_grad():
                latent_sample, latent_mean, latent_logvar, hidden_state = self.encode_running_trajectory()
            
            # add this initial hidden state to the policy storage
            assert len(self.policy_storage.latent_mean) == 0  # make sure we emptied buffers
            self.policy_storage.hidden_states[0].copy_(hidden_state)
            self.policy_storage.latent_samples.append(latent_sample.clone())
            self.policy_storage.latent_mean.append(latent_mean.clone())
            self.policy_storage.latent_logvar.append(latent_logvar.clone())
            
            # rollout policies for a few steps
            for step in range(self.args.policy_num_steps):
                # sample actions from policy
                with torch.no_grad():
                    value, action = utl.select_action(
                        policy=self.policy,
                        state=prev_state,
                        belief=belief,
                        deterministic=False,
                        latent_sample=latent_sample,
                        latent_mean=latent_mean,
                        latent_logvar=latent_logvar,
                    )
                
                # take step in the environment
                [next_state, belief], (rew_raw, rew_normalised), terminated, truncated, infos = utl.env_step(self.envs, action, self.args)
                done = torch.as_tensor(np.logical_or(terminated, truncated), device=device, dtype=torch.float32).view(-1, 1)
                
                # create mask for episode ends
                masks_done = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done]).to(device)
                
                with torch.no_grad():
                    # compute next embedding (for next loop and/or value prediction bootstrap)
                    latent_sample, latent_mean, latent_logvar, hidden_state = utl.update_encoding(
                        encoder=self.vae.encoder,
                        next_obs=next_state,
                        action=action,
                        reward=rew_raw,
                        done=done,
                        hidden_state=hidden_state)
                
                # before resetting, update the embedding and add to vae buffer
                # (last state might include useful task info)
                self.vae.rollout_storage.insert(action.detach().clone(),
                                                next_state.clone(),
                                                rew_raw.clone(),
                                                done.clone())
                
                # reset environments that are done
                done_indices = np.argwhere(done.cpu().flatten()).flatten()
                if len(done_indices) > 0:
                    next_state, belief = utl.reset_env(self.envs, self.args, indices=done_indices, state=next_state)
                
                # add experience to policy buffer
                self.policy_storage.insert(
                    state=next_state,
                    belief=belief,
                    actions=action,
                    rewards_raw=rew_raw,
                    rewards_normalised=rew_normalised,
                    value_preds=value,
                    masks=masks_done,
                    hidden_states=hidden_state.squeeze(0),
                    latent_sample=latent_sample,
                    latent_mean=latent_mean,
                    latent_logvar=latent_logvar,
                )
                
                prev_state = next_state
                self.frames += self.args.num_processes
            
            # --- UPDATE ---
            if self.args.precollect_len <= self.frames:
                # check if we are pre-training the VAE
                if self.args.pretrain_len > self.iter_idx:
                    for p in range(self.args.num_vae_updates_per_pretrain):
                        self.vae.compute_vae_loss(update=True, pretrain_index=self.iter_idx * self.args.num_vae_updates_per_pretrain + p)
                else:
                    train_stats = self.update(state=prev_state,
                                                belief=belief,
                                                latent_sample=latent_sample,
                                                latent_mean=latent_mean,
                                                latent_logvar=latent_logvar)
                    # log
                    run_stats = [action, self.policy_storage.action_log_probs, value]
                    with torch.no_grad():
                        self.log(run_stats, train_stats, start_time)
            
            # clean up after update
            self.policy_storage.after_update()
        
        self.envs.close()
    
    def encode_running_trajectory(self):
        """
        (Re-)Encodes (for each process) the entire current trajectory.
        Returns sample/mean/logvar and hidden state (if applicable) for the current timestep.
        :return:
        """
        
        # for each process, get the current batch (zero-padded obs/act/rew + length indicators)
        next_obs, act, rew, lens = self.vae.rollout_storage.get_running_batch()
        
        # get embedding - will return (1+sequence_len) * batch * input_size -- includes the prior!
        all_latent_samples, all_latent_means, all_latent_logvars, all_hidden_states = self.vae.encoder(actions=act,
                                                                                                        states=next_obs,
                                                                                                        rewards=rew,
                                                                                                        hidden_state=None,
                                                                                                        return_prior=True
                                                                                                        )
        
        # get the embedding / hidden state of the current time step (need to do this since we zero-padded)
        latent_sample = (torch.stack([all_latent_samples[lens[i]][i] for i in range(len(lens))])).to(device)
        latent_mean = (torch.stack([all_latent_means[lens[i]][i] for i in range(len(lens))])).to(device)
        latent_logvar = (torch.stack([all_latent_logvars[lens[i]][i] for i in range(len(lens))])).to(device)
        hidden_state = (torch.stack([all_hidden_states[lens[i]][i] for i in range(len(lens))])).to(device)
        
        return latent_sample, latent_mean, latent_logvar, hidden_state
    
    def get_value(self, state, belief, latent_sample, latent_mean, latent_logvar):
        latent = utl.get_latent_for_policy(latent_sample=latent_sample, latent_mean=latent_mean, latent_logvar=latent_logvar)
        return self.policy.actor_critic.get_value(state=state, belief=belief, latent=latent).detach()
    
    def update(self, state, belief, latent_sample, latent_mean, latent_logvar):
        """
        Meta-update.
        Here the policy is updated for good average performance across tasks.
        :return:
        """
        # update policy (if we are not pre-training, have enough data in the vae buffer, and are not at iteration 0)
        if self.iter_idx >= self.args.pretrain_len and self.iter_idx > 0:
            
            # bootstrap next value prediction
            with torch.no_grad():
                next_value = self.get_value(state=state, belief=belief, latent_sample=latent_sample, latent_mean=latent_mean, latent_logvar=latent_logvar)
            
            # Use generalized advantage estimation (GAE) for ppo policy returns.
            self.policy_storage.compute_returns(next_value, self.args.policy_gamma, self.args.policy_tau)
            
            # update agent (this will also call the VAE update!)
            policy_train_stats = self.policy.update(
                policy_storage=self.policy_storage,
                compute_vae_loss=self.vae.compute_vae_loss)
        else:
            policy_train_stats = 0, 0, 0, 0
            
            # pre-train the VAE
            if self.iter_idx < self.args.pretrain_len:
                self.vae.compute_vae_loss(update=True)
        
        return policy_train_stats
    
    def log(self, run_stats, train_stats, start_time):
        # --- visualise behaviour of policy ---
        if (self.iter_idx + 1) % self.args.vis_interval == 0:
            ret_rms = self.envs.venv.ret_rms
            utl_eval.visualise_behaviour(args=self.args,
                                            policy=self.policy,
                                            image_folder=self.logger.full_output_folder,
                                            iter_idx=self.iter_idx,
                                            ret_rms=ret_rms,
                                            encoder=self.vae.encoder,
                                            reward_decoder=self.vae.reward_decoder,
                                            tasks=None,
                                        )
        
        # --- evaluate policy ----
        if (self.iter_idx + 1) % self.args.eval_interval == 0:
            ret_rms = self.envs.venv.ret_rms
            returns_per_episode = utl_eval.evaluate(args=self.args,
                                                    policy=self.policy,
                                                    ret_rms=ret_rms,
                                                    encoder=self.vae.encoder,
                                                    iter_idx=self.iter_idx,
                                                    tasks=None,
                                                    )
            
            # log the return avg/std across tasks (=processes)
            returns_avg = returns_per_episode.mean(dim=0)
            returns_std = returns_per_episode.std(dim=0)
            for k in range(len(returns_avg)):
                self.logger.add('return_avg_per_iter/episode_{}'.format(k + 1), returns_avg[k], self.iter_idx)
                self.logger.add('return_avg_per_frame/episode_{}'.format(k + 1), returns_avg[k], self.frames)
                self.logger.add('return_std_per_iter/episode_{}'.format(k + 1), returns_std[k], self.iter_idx)
                self.logger.add('return_std_per_frame/episode_{}'.format(k + 1), returns_std[k], self.frames)
            
            print(f"Updates {self.iter_idx}, "
                    f"Frames {self.frames}, "
                    f"FPS {int(self.frames / (time.time() - start_time))}, "
                    f"\n Mean return (train): {returns_avg[-1].item()} \n"
                )
        
        # --- save models ---
        if (self.iter_idx + 1) % self.args.save_interval == 0:
            save_path = os.path.join(self.logger.full_output_folder, 'models')
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            
            idx_labels = ['']
            if self.args.save_intermediate_models:
                idx_labels.append(int(self.iter_idx))
            
            for idx_label in idx_labels:
                torch.save(self.policy.actor_critic, os.path.join(save_path, f"policy{idx_label}.pt"))
                torch.save(self.vae.encoder, os.path.join(save_path, f"encoder{idx_label}.pt"))
                torch.save(self.vae.reward_decoder, os.path.join(save_path, f"reward_decoder{idx_label}.pt"))
                # save normalisation params of envs
                rew_rms = self.envs.venv.ret_rms
                utl.save_obj(rew_rms, save_path, f"env_rew_rms{idx_label}")
        
        # --- log some other things ---
        if ((self.iter_idx + 1) % self.args.log_interval == 0) and (train_stats is not None):
            
            self.logger.add('environment/state_max', self.policy_storage.prev_state.max(), self.iter_idx)
            self.logger.add('environment/state_min', self.policy_storage.prev_state.min(), self.iter_idx)
            self.logger.add('environment/rew_max', self.policy_storage.rewards_raw.max(), self.iter_idx)
            self.logger.add('environment/rew_min', self.policy_storage.rewards_raw.min(), self.iter_idx)
            self.logger.add('policy_losses/value_loss', train_stats[0], self.iter_idx)
            self.logger.add('policy_losses/action_loss', train_stats[1], self.iter_idx)
            self.logger.add('policy_losses/dist_entropy', train_stats[2], self.iter_idx)
            self.logger.add('policy_losses/sum', train_stats[3], self.iter_idx)
            self.logger.add('policy/action', run_stats[0][0].float().mean(), self.iter_idx)
            if hasattr(self.policy.actor_critic, 'logstd'):
                self.logger.add('policy/action_logstd', self.policy.actor_critic.dist.logstd.mean(), self.iter_idx)
            self.logger.add('policy/action_logprob', run_stats[1].mean(), self.iter_idx)
            self.logger.add('policy/value', run_stats[2].mean(), self.iter_idx)
            self.logger.add('encoder/latent_mean', torch.cat(self.policy_storage.latent_mean).mean(), self.iter_idx)
            self.logger.add('encoder/latent_logvar', torch.cat(self.policy_storage.latent_logvar).mean(), self.iter_idx)
            # log the average weights and gradients of all models (where applicable)
            for [model, name] in [
                [self.policy.actor_critic, 'policy'],
                [self.vae.encoder, 'encoder'],
                [self.vae.reward_decoder, 'reward_decoder'],
            ]:
                if model is not None:
                    param_list = list(model.parameters())
                    param_mean = np.mean([param_list[i].data.cpu().numpy().mean() for i in range(len(param_list))])
                    self.logger.add('weights/{}'.format(name), param_mean, self.iter_idx)
                    if name == 'policy':
                        self.logger.add('weights/policy_std', param_list[0].data.mean(), self.iter_idx)
                    if param_list[0].grad is not None:
                        param_grad_mean = np.mean([param_list[i].grad.cpu().numpy().mean() for i in range(len(param_list))])
                        self.logger.add('gradients/{}'.format(name), param_grad_mean, self.iter_idx)
