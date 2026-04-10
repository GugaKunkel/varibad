import matplotlib.pyplot as plt
import numpy as np
import torch

from environments.parallel_envs import make_vec_envs
from utils import helpers as utl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def evaluate(args,
             policy,
             ret_rms,
             iter_idx,
             tasks,
             encoder,
             num_episodes=None
             ):
    env_name = args.env_name
    if hasattr(args, 'test_env_name'):
        env_name = args.test_env_name
    if num_episodes is None:
        num_episodes = args.max_rollouts_per_task
    num_processes = args.num_processes

    # --- set up the things we want to log ---

    # for each process, we log the returns during the first, second, ... episode
    # (such that we have a minimum of [num_episodes]; the last column is for
    #  any overflow and will be discarded at the end, because we need to wait until
    #  all processes have at least [num_episodes] many episodes)
    returns_per_episode = torch.zeros((num_processes, num_episodes + 1)).to(device)

    # --- initialise environments and latents ---

    envs = make_vec_envs(env_name,
                         seed=args.seed * 42 + iter_idx,
                         num_processes=num_processes,
                         gamma=args.policy_gamma,
                         device=device,
                         rank_offset=num_processes + 1,  # to use diff tmp folders than main processes
                         episodes_per_task=num_episodes,
                         normalise_rew=True,
                         ret_rms=ret_rms,
                         tasks=tasks,
                         add_done_info=args.max_rollouts_per_task > 1,
                         )
    num_steps = envs._max_episode_steps

    # reset environments
    state, belief = utl.reset_env(envs, args)

    # this counts how often an agent has done the same task already
    task_count = torch.zeros(num_processes).long().to(device)
    # reset latent state to prior
    latent_sample, latent_mean, latent_logvar, hidden_state = encoder.prior(num_processes)

    for episode_idx in range(num_episodes):

        for step_idx in range(num_steps):

            with torch.no_grad():
                _, action = utl.select_action(policy=policy,
                                              state=state,
                                              belief=belief,
                                              latent_sample=latent_sample,
                                              latent_mean=latent_mean,
                                              latent_logvar=latent_logvar,
                                              deterministic=True)

            # observe reward and next obs
            [state, belief], (rew_raw, rew_normalised), terminated, truncated, infos = utl.env_step(envs, action, args)
            done = np.logical_or(terminated, truncated)
            done_mdp = [info['done_mdp'] for info in infos]

            # update the hidden state
            latent_sample, latent_mean, latent_logvar, hidden_state = utl.update_encoding(encoder=encoder,
                                                                                            next_obs=state,
                                                                                            action=action,
                                                                                            reward=rew_raw,
                                                                                            done=None,
                                                                                            hidden_state=hidden_state)

            # add rewards
            returns_per_episode[range(num_processes), task_count] += rew_raw.view(-1)

            for i in np.argwhere(done_mdp).flatten():
                # count task up, but cap at num_episodes + 1
                task_count[i] = min(task_count[i] + 1, num_episodes)  # zero-indexed, so no +1
            if np.sum(done) > 0:
                done_indices = np.argwhere(done.flatten()).flatten()
                state, belief = utl.reset_env(envs, args, indices=done_indices, state=state)

    envs.close()

    return returns_per_episode[:, :num_episodes]


def visualise_behaviour(args,
                        policy,
                        image_folder,
                        iter_idx,
                        ret_rms,
                        tasks,
                        encoder=None,
                        reward_decoder=None,
                        ):
    # initialise environment
    env = make_vec_envs(env_name=args.env_name,
                        seed=args.seed * 42 + iter_idx,
                        num_processes=1,
                        gamma=args.policy_gamma,
                        device=device,
                        episodes_per_task=args.max_rollouts_per_task,
                        normalise_rew=True, ret_rms=ret_rms,
                        rank_offset=args.num_processes + 42,  # not sure if the temp folders would otherwise clash
                        tasks=tasks
                        )
    # get a sample rollout
    unwrapped_env = env.venv.unwrapped.envs[0]
    if hasattr(env.venv.unwrapped.envs[0], 'unwrapped'):
        unwrapped_env = unwrapped_env.unwrapped
    if hasattr(unwrapped_env, 'visualise_behaviour'):
        # if possible, get it from the env directly
        # (this might visualise other things in addition)
        traj = unwrapped_env.visualise_behaviour(env=env,
                                                 args=args,
                                                 policy=policy,
                                                 iter_idx=iter_idx,
                                                 encoder=encoder,
                                                 reward_decoder=reward_decoder,
                                                 image_folder=image_folder,
                                                 )
    else:
        traj = get_test_rollout(args, env, policy, encoder)

    if len(traj) == 7:
        latent_means, latent_logvars, episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, _ = traj
    else:
        latent_means, latent_logvars, episode_prev_obs, episode_next_obs, episode_actions, episode_rewards = traj

    if latent_means is not None:
        plot_latents(latent_means, latent_logvars,
                     image_folder=image_folder,
                     iter_idx=iter_idx
                     )

    env.close()


def get_test_rollout(args, env, policy, encoder=None):
    num_episodes = args.max_rollouts_per_task

    # --- initialise things we want to keep track of ---

    episode_prev_obs = [[] for _ in range(num_episodes)]
    episode_next_obs = [[] for _ in range(num_episodes)]
    episode_actions = [[] for _ in range(num_episodes)]
    episode_rewards = [[] for _ in range(num_episodes)]

    episode_lengths = []

    if encoder is not None:
        episode_latent_samples = [[] for _ in range(num_episodes)]
        episode_latent_means = [[] for _ in range(num_episodes)]
        episode_latent_logvars = [[] for _ in range(num_episodes)]
    else:
        curr_latent_sample = curr_latent_mean = curr_latent_logvar = None
        episode_latent_means = episode_latent_logvars = None

    # --- roll out policy ---

    # (re)set environment
    env.unwrapped.reset_task()
    state, belief = utl.reset_env(env, args)
    state = state.reshape((1, -1)).to(device)

    for episode_idx in range(num_episodes):
        if encoder is not None:
            if episode_idx == 0:
                # reset to prior
                curr_latent_sample, curr_latent_mean, curr_latent_logvar, hidden_state = encoder.prior(1)
                curr_latent_sample = curr_latent_sample[0].to(device)
                curr_latent_mean = curr_latent_mean[0].to(device)
                curr_latent_logvar = curr_latent_logvar[0].to(device)
            episode_latent_samples[episode_idx].append(curr_latent_sample[0].clone())
            episode_latent_means[episode_idx].append(curr_latent_mean[0].clone())
            episode_latent_logvars[episode_idx].append(curr_latent_logvar[0].clone())

        for step_idx in range(1, env._max_episode_steps + 1):

            episode_prev_obs[episode_idx].append(state.clone())

            latent = utl.get_latent_for_policy(latent_sample=curr_latent_sample,
                                               latent_mean=curr_latent_mean,
                                               latent_logvar=curr_latent_logvar)
            _, action = policy.act(state=state.view(-1), latent=latent, belief=belief, deterministic=True)
            action = action.reshape((1, *action.shape))

            # observe reward and next obs
            [state, belief], (rew_raw, rew_normalised), terminated, truncated, infos = utl.env_step(env, action, args)
            done = np.logical_or(terminated, truncated)
            state = state.reshape((1, -1)).to(device)

            if encoder is not None:
                # update task embedding
                curr_latent_sample, curr_latent_mean, curr_latent_logvar, hidden_state = encoder(
                    action.float().to(device),
                    state,
                    rew_raw.reshape((1, 1)).float().to(device),
                    hidden_state,
                    return_prior=False)

                episode_latent_samples[episode_idx].append(curr_latent_sample[0].clone())
                episode_latent_means[episode_idx].append(curr_latent_mean[0].clone())
                episode_latent_logvars[episode_idx].append(curr_latent_logvar[0].clone())

            episode_next_obs[episode_idx].append(state.clone())
            episode_rewards[episode_idx].append(rew_raw.clone())
            episode_actions[episode_idx].append(action.clone())

            if infos[0]['done_mdp']:
                break
        episode_lengths.append(step_idx)

    # clean up
    if encoder is not None:
        episode_latent_means = [torch.stack(e) for e in episode_latent_means]
        episode_latent_logvars = [torch.stack(e) for e in episode_latent_logvars]

    episode_prev_obs = [torch.cat(e) for e in episode_prev_obs]
    episode_next_obs = [torch.cat(e) for e in episode_next_obs]
    episode_actions = [torch.cat(e) for e in episode_actions]
    episode_rewards = [torch.cat(r) for r in episode_rewards]

    return episode_latent_means, episode_latent_logvars, episode_prev_obs, episode_next_obs, episode_actions, episode_rewards


def plot_latents(latent_means,
                 latent_logvars,
                 image_folder,
                 iter_idx,
                 ):
    """
    Plot mean/variance over time
    """

    num_rollouts = len(latent_means)
    num_episode_steps = len(latent_means[0])

    latent_means = torch.cat(latent_means).cpu().detach().numpy()
    latent_logvars = torch.cat(latent_logvars).cpu().detach().numpy()

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(latent_means.shape[0]), latent_means, '-', alpha=0.5)
    plt.plot(range(latent_means.shape[0]), latent_means.mean(axis=1), 'k-')
    for tj in np.cumsum([0, *[num_episode_steps for _ in range(num_rollouts)]]):
        span = latent_means.max() - latent_means.min()
        plt.plot([tj + 0.5, tj + 0.5],
                 [latent_means.min() - span * 0.05, latent_means.max() + span * 0.05],
                 'k--', alpha=0.5)
    plt.xlabel('env steps', fontsize=15)
    plt.ylabel('latent mean', fontsize=15)

    plt.subplot(1, 2, 2)
    latent_var = np.exp(latent_logvars)
    plt.plot(range(latent_logvars.shape[0]), latent_var, '-', alpha=0.5)
    plt.plot(range(latent_logvars.shape[0]), latent_var.mean(axis=1), 'k-')
    for tj in np.cumsum([0, *[num_episode_steps for _ in range(num_rollouts)]]):
        span = latent_var.max() - latent_var.min()
        plt.plot([tj + 0.5, tj + 0.5],
                 [latent_var.min() - span * 0.05, latent_var.max() + span * 0.05],
                 'k--', alpha=0.5)
    plt.xlabel('env steps', fontsize=15)
    plt.ylabel('latent variance', fontsize=15)

    plt.tight_layout()
    if image_folder is not None:
        plt.savefig('{}/{}_latents'.format(image_folder, iter_idx))
        plt.close()
    else:
        plt.show()
