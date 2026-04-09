import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RolloutStorageVAE(object):
    def __init__(self, num_processes, max_trajectory_len, max_num_rollouts, state_dim, action_dim):
        """ Store everything that is needed for the VAE update """
        
        self.max_buffer_size = max_num_rollouts  # maximum buffer len (number of trajectories)
        self.insert_idx = 0  # at which index we're currently inserting new data
        self.buffer_len = 0  # how much of the buffer has been filled
        self.max_traj_len = max_trajectory_len # how long a trajectory can be at max (horizon)
        
        # buffers for completed rollouts (stored on CPU)
        if self.max_buffer_size > 0:
            self.next_state = torch.zeros((self.max_traj_len, self.max_buffer_size, state_dim))
            self.actions = torch.zeros((self.max_traj_len, self.max_buffer_size, action_dim))
            self.rewards = torch.zeros((self.max_traj_len, self.max_buffer_size, 1))
            self.trajectory_lens = [0] * self.max_buffer_size
        
        # storage for each running process (stored on GPU)
        self.num_processes = num_processes
        self.curr_timestep = torch.zeros((num_processes)).long()  # count environment steps so we know where to insert
        self.running_next_state = torch.zeros((self.max_traj_len, num_processes, state_dim)).to(device)  # for each episode will have obs 1...N
        self.running_rewards = torch.zeros((self.max_traj_len, num_processes, 1)).to(device)
        self.running_actions = torch.zeros((self.max_traj_len, num_processes, action_dim)).to(device)
    
    def get_running_batch(self):
        return self.running_next_state, self.running_actions, self.running_rewards, self.curr_timestep
    
    def insert(self, actions, next_state, rewards, done):
        
        # add to temporary buffer
        already_inserted = False
        if len(np.unique(self.curr_timestep)) == 1:
            self.running_next_state[self.curr_timestep[0]] = next_state
            self.running_rewards[self.curr_timestep[0]] = rewards
            self.running_actions[self.curr_timestep[0]] = actions
            self.curr_timestep += 1
            already_inserted = True
        
        already_reset = False
        if done.sum() == self.num_processes:  # check if we can process the entire batch at once
            
            # add to permanent (up to max_buffer_len) buffer
            if self.max_buffer_size > 0:
                # check where to insert data
                if self.insert_idx + self.num_processes > self.max_buffer_size:
                    # keep track of how much we filled the buffer (for sampling from it)
                    self.buffer_len = self.insert_idx
                    # this will keep some entries at the end of the buffer without overwriting them,
                    # but the buffer is large enough to make this negligible
                    self.insert_idx = 0
                else:
                    self.buffer_len = max(self.buffer_len, self.insert_idx)
                # add; note: num trajectories are along dim=1,
                # trajectory length along dim=0, to match pytorch RNN interface
                self.next_state[:, self.insert_idx:self.insert_idx + self.num_processes] = self.running_next_state
                self.actions[:, self.insert_idx:self.insert_idx+self.num_processes] = self.running_actions
                self.rewards[:, self.insert_idx:self.insert_idx+self.num_processes] = self.running_rewards
                self.trajectory_lens[self.insert_idx:self.insert_idx+self.num_processes] = self.curr_timestep.clone()
                self.insert_idx += self.num_processes
            
            # empty running buffer
            self.running_next_state *= 0
            self.running_rewards *= 0
            self.running_actions *= 0
            self.curr_timestep *= 0
            already_reset = True
        
        if (not already_inserted) or (not already_reset):
            for i in range(self.num_processes):
                if not already_inserted:
                    self.running_next_state[self.curr_timestep[i], i] = next_state[i]
                    self.running_rewards[self.curr_timestep[i], i] = rewards[i]
                    self.running_actions[self.curr_timestep[i], i] = actions[i]
                    self.curr_timestep[i] += 1
                if not already_reset:
                    # if we are at the end of a task, dump the data into the larger buffer
                    if done[i]:
                        # add to permanent (up to max_buffer_len) buffer
                        if self.max_buffer_size > 0:
                            # check where to insert data
                            if self.insert_idx + 1 > self.max_buffer_size:
                                # keep track of how much we filled the buffer (for sampling from it)
                                self.buffer_len = self.insert_idx
                                # this will keep some entries at the end of the buffer without overwriting them,
                                # but the buffer is large enough to make this negligible
                                self.insert_idx = 0
                            else:
                                self.buffer_len = max(self.buffer_len, self.insert_idx)
                            # add; note: num trajectories are along dim=1,
                            # trajectory length along dim=0, to match pytorch RNN interface
                            self.next_state[:, self.insert_idx] = self.running_next_state[:, i].to('cpu')
                            self.actions[:, self.insert_idx] = self.running_actions[:, i].to('cpu')
                            self.rewards[:, self.insert_idx] = self.running_rewards[:, i].to('cpu')
                            self.trajectory_lens[self.insert_idx] = self.curr_timestep[i].clone()
                            self.insert_idx += 1
                        # empty running buffer
                        self.running_next_state[:, i] *= 0
                        self.running_rewards[:, i] *= 0
                        self.running_actions[:, i] *= 0
                        self.curr_timestep[i] = 0
    
    def ready_for_update(self):
        return len(self) > 0
    
    def __len__(self):
        return self.buffer_len
    
    def get_batch(self, batchsize=5, replace=False):
        batchsize = min(self.buffer_len, batchsize)
        
        # select the indices for the processes from which we pick
        rollout_indices = np.random.choice(range(self.buffer_len), batchsize, replace=replace)
        # trajectory length of the individual rollouts we picked
        trajectory_lens = np.array(self.trajectory_lens)[rollout_indices]
        
        # select the rollouts we want
        next_obs = self.next_state[:, rollout_indices, :]
        actions = self.actions[:, rollout_indices, :]
        rewards = self.rewards[:, rollout_indices, :]
        return next_obs.to(device), actions.to(device), rewards.to(device), trajectory_lens
