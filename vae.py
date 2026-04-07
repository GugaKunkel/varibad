import warnings

import gymnasium as gym
import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn

from models.decoder import RewardDecoder
from models.encoder import RNNEncoder
from utils.storage_vae import RolloutStorageVAE

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class VaribadVAE:
    """
    VAE of VariBAD:
    - has an encoder and decoder
    - can compute the ELBO loss
    - can update the VAE (encoder+decoder)
    """

    def __init__(self, args, logger, get_iter_idx):

        self.args = args
        self.logger = logger
        self.get_iter_idx = get_iter_idx

        # initialise the encoder
        self.encoder = RNNEncoder(
            args=self.args,
            hidden_size=self.args.encoder_gru_hidden_size,
            latent_dim=self.args.latent_dim,
            action_dim=self.args.action_dim,
            action_embed_dim=self.args.action_embedding_size,
            state_dim=self.args.state_dim,
            state_embed_dim=self.args.state_embedding_size,
            reward_size=1,
            reward_embed_size=self.args.reward_embedding_size,
        ).to(device)


        # initialise reward decoder
        self.reward_decoder = RewardDecoder(
            layers=self.args.reward_decoder_layers,
            latent_dim=self.args.latent_dim,
            num_states=self.args.num_states,
        ).to(device)

        # initialise rollout storage for the VAE update
        # (this differs from the data that the on-policy RL algorithm uses)
        self.rollout_storage = RolloutStorageVAE(num_processes=self.args.num_processes,
                                                 max_trajectory_len=self.args.max_trajectory_len,
                                                 zero_pad=True,
                                                 max_num_rollouts=self.args.size_vae_buffer,
                                                 state_dim=self.args.state_dim,
                                                 action_dim=self.args.action_dim
                                                 )

        # initalise optimiser for the encoder and decoders
        self.optimiser_vae = torch.optim.Adam([*self.encoder.parameters(), *self.reward_decoder.parameters()], lr=self.args.lr_vae)

    def compute_rew_reconstruction_loss(self, latent, next_obs, reward, return_predictions=False):
        """ Compute reward reconstruction loss.
        (No reduction of loss along batch dimension is done here; sum/avg has to be done outside) """
        rew_pred = self.reward_decoder(latent)
        if self.args.rew_pred_type == 'categorical':
            rew_pred = F.softmax(rew_pred, dim=-1)
        elif self.args.rew_pred_type == 'bernoulli':
            rew_pred = torch.sigmoid(rew_pred)

        env = gym.make(self.args.env_name)
        env_task = env.unwrapped if hasattr(env, 'unwrapped') else env
        state_indices = env_task.task_to_id(next_obs).to(device)
        if state_indices.dim() < rew_pred.dim():
            state_indices = state_indices.unsqueeze(-1)
        rew_pred = rew_pred.gather(dim=-1, index=state_indices)
        rew_target = (reward == 1).float()
        if self.args.rew_pred_type in ['categorical', 'bernoulli']:
            loss_rew = F.binary_cross_entropy(rew_pred, rew_target, reduction='none').mean(dim=-1)
        else:
            raise NotImplementedError

        if return_predictions:
            return loss_rew, rew_pred
        else:
            return loss_rew

    def compute_kl_loss(self, latent_mean, latent_logvar, elbo_indices):
        # -- KL divergence
        gauss_dim = latent_mean.shape[-1]
        # add the gaussian prior
        all_means = torch.cat((torch.zeros(1, *latent_mean.shape[1:]).to(device), latent_mean))
        all_logvars = torch.cat((torch.zeros(1, *latent_logvar.shape[1:]).to(device), latent_logvar))
        # https://arxiv.org/pdf/1811.09975.pdf
        # KL(N(mu,E)||N(m,S)) = 0.5 * (log(|S|/|E|) - K + tr(S^-1 E) + (m-mu)^T S^-1 (m-mu)))
        mu = all_means[1:]
        m = all_means[:-1]
        logE = all_logvars[1:]
        logS = all_logvars[:-1]
        kl_divergences = 0.5 * (torch.sum(logS, dim=-1) - torch.sum(logE, dim=-1) - gauss_dim + torch.sum(
            1 / torch.exp(logS) * torch.exp(logE), dim=-1) + ((m - mu) / torch.exp(logS) * (m - mu)).sum(dim=-1))

        # returns, for each ELBO_t term, one KL (so H+1 kl's)
        if elbo_indices is not None:
            batchsize = kl_divergences.shape[-1]
            task_indices = torch.arange(batchsize).repeat(self.args.vae_subsample_elbos)
            kl_divergences = kl_divergences[elbo_indices, task_indices].reshape((self.args.vae_subsample_elbos, batchsize))

        return kl_divergences

    def compute_loss(self, latent_mean, latent_logvar, vae_prev_obs, vae_next_obs, vae_actions, vae_rewards, trajectory_lens):
        """
        Computes the VAE loss for the given data.
        Batches everything together and therefore needs all trajectories to be of the same length.
        (Important because we need to separate ELBOs and decoding terms so can't collapse those dimensions)
        """

        num_unique_trajectory_lens = len(np.unique(trajectory_lens))

        assert (num_unique_trajectory_lens == 1) or (self.args.vae_subsample_elbos is not None)

        # cut down the batch to the longest trajectory length
        # this way we can preserve the structure
        # but we will waste some computation on zero-padded trajectories that are shorter than max_traj_len
        max_traj_len = np.max(trajectory_lens)
        latent_mean = latent_mean[:max_traj_len + 1]
        latent_logvar = latent_logvar[:max_traj_len + 1]
        vae_prev_obs = vae_prev_obs[:max_traj_len]
        vae_next_obs = vae_next_obs[:max_traj_len]
        vae_actions = vae_actions[:max_traj_len]
        vae_rewards = vae_rewards[:max_traj_len]

        # take one sample for each ELBO term
        latent_samples = self.encoder._sample_gaussian(latent_mean, latent_logvar)

        num_elbos = latent_samples.shape[0]
        num_decodes = vae_prev_obs.shape[0]
        batchsize = latent_samples.shape[1]  # number of trajectories

        # subsample elbo terms
        #   shape before: num_elbos * batchsize * dim
        #   shape after: vae_subsample_elbos * batchsize * dim
        if self.args.vae_subsample_elbos is not None:
            # randomly choose which elbo's to subsample
            if num_unique_trajectory_lens == 1:
                elbo_indices = torch.LongTensor(self.args.vae_subsample_elbos * batchsize).random_(0, num_elbos)    # select diff elbos for each task
            else:
                # if we have different trajectory lengths, subsample elbo indices separately
                # up to their maximum possible encoding length;
                # only allow duplicates if the sample size would be larger than the number of samples
                elbo_indices = np.concatenate([np.random.choice(range(0, t + 1), self.args.vae_subsample_elbos,
                                                                replace=self.args.vae_subsample_elbos > (t+1)) for t in trajectory_lens])
                if max_traj_len < self.args.vae_subsample_elbos:
                    warnings.warn('The required number of ELBOs is larger than the shortest trajectory, so there will be duplicates in your batch.')
            task_indices = torch.arange(batchsize).repeat(self.args.vae_subsample_elbos)  # for selection mask
            latent_samples = latent_samples[elbo_indices, task_indices, :].reshape((self.args.vae_subsample_elbos, batchsize, -1))
            num_elbos = latent_samples.shape[0]
        else:
            elbo_indices = None

        # expand the state/rew/action inputs to the decoder (to match size of latents)
        # shape will be: [num tasks in batch] x [num elbos] x [len trajectory (reconstrution loss)] x [dimension]
        dec_next_obs = vae_next_obs.unsqueeze(0).expand((num_elbos, *vae_next_obs.shape))
        dec_rewards = vae_rewards.unsqueeze(0).expand((num_elbos, *vae_rewards.shape))

        # expand the latent (to match the number of state/rew/action inputs to the decoder)
        # shape will be: [num tasks in batch] x [num elbos] x [len trajectory (reconstrution loss)] x [dimension]
        dec_embedding = latent_samples.unsqueeze(0).expand((num_decodes, *latent_samples.shape)).transpose(1, 0)

        # compute reconstruction loss for this trajectory (for each timestep that was encoded, decode everything and sum it up)
        # shape: [num_elbo_terms] x [num_reconstruction_terms] x [num_trajectories]
        rew_reconstruction_loss = self.compute_rew_reconstruction_loss(dec_embedding, dec_next_obs, dec_rewards)
        # avg/sum across individual ELBO terms
        if self.args.vae_avg_elbo_terms:
            rew_reconstruction_loss = rew_reconstruction_loss.mean(dim=0)
        else:
            rew_reconstruction_loss = rew_reconstruction_loss.sum(dim=0)
        # avg/sum across individual reconstruction terms
        if self.args.vae_avg_reconstruction_terms:
            rew_reconstruction_loss = rew_reconstruction_loss.mean(dim=0)
        else:
            rew_reconstruction_loss = rew_reconstruction_loss.sum(dim=0)
        # average across tasks
        rew_reconstruction_loss = rew_reconstruction_loss.mean()

        # compute the KL term for each ELBO term of the current trajectory
        # shape: [num_elbo_terms] x [num_trajectories]
        kl_loss = self.compute_kl_loss(latent_mean, latent_logvar, elbo_indices)
        # avg/sum the elbos
        if self.args.vae_avg_elbo_terms:
            kl_loss = kl_loss.mean(dim=0)
        else:
            kl_loss = kl_loss.sum(dim=0)
        # average across tasks
        kl_loss = kl_loss.sum(dim=0).mean()

        return rew_reconstruction_loss, kl_loss

    def compute_vae_loss(self, update=False, pretrain_index=None):
        """ Returns the VAE loss """

        if not self.rollout_storage.ready_for_update():
            return 0

        # get a mini-batch
        vae_prev_obs, vae_next_obs, vae_actions, vae_rewards, trajectory_lens = self.rollout_storage.get_batch(batchsize=self.args.vae_batch_num_trajs)
        # vae_prev_obs will be of size: max trajectory len x num trajectories x dimension of observations

        # pass through encoder (outputs will be: (max_traj_len+1) x number of rollouts x latent_dim -- includes the prior!)
        _, latent_mean, latent_logvar, _ = self.encoder(actions=vae_actions,
                                                        states=vae_next_obs,
                                                        rewards=vae_rewards,
                                                        hidden_state=None,
                                                        return_prior=True,
                                                        detach_every=self.args.tbptt_stepsize if hasattr(self.args, 'tbptt_stepsize') else None,
                                                        )
        rew_reconstruction_loss, kl_loss = self.compute_loss(
            latent_mean, latent_logvar, vae_prev_obs, vae_next_obs, vae_actions, vae_rewards, trajectory_lens
        )

        # VAE loss = KL loss + reward reconstruction
        # take average (this is the expectation over p(M))
        loss = (rew_reconstruction_loss + self.args.kl_weight * kl_loss).mean()

        # make sure we can compute gradients
        assert kl_loss.requires_grad
        assert rew_reconstruction_loss.requires_grad

        # overall loss
        elbo_loss = loss.mean()

        if update:
            self.optimiser_vae.zero_grad()
            elbo_loss.backward()
            # clip gradients
            if self.args.encoder_max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.encoder.parameters(), self.args.encoder_max_grad_norm)
            if self.args.decoder_max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.reward_decoder.parameters(), self.args.decoder_max_grad_norm)
            # update
            self.optimiser_vae.step()

        self.log(elbo_loss, rew_reconstruction_loss, kl_loss, pretrain_index)
        return elbo_loss

    def log(self, elbo_loss, rew_reconstruction_loss, kl_loss,
            pretrain_index=None):

        if pretrain_index is None:
            curr_iter_idx = self.get_iter_idx()
        else:
            curr_iter_idx = - self.args.pretrain_len * self.args.num_vae_updates_per_pretrain + pretrain_index

        if curr_iter_idx % self.args.log_interval == 0:
            self.logger.add('vae_losses/reward_reconstr_err', rew_reconstruction_loss.mean(), curr_iter_idx)
            self.logger.add('vae_losses/kl', kl_loss.mean(), curr_iter_idx)
            self.logger.add('vae_losses/sum', elbo_loss, curr_iter_idx)
