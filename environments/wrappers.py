import gymnasium as gym
import numpy as np
from gymnasium import spaces

class VariBadWrapper(gym.Wrapper):
    def __init__(self,
                env,
                episodes_per_task,
                add_done_info=None,  # force to turn this on/off
                ):
        """
        Wrapper, creates a multi-episode (BA)MDP around a one-episode MDP. Automatically deals with
        - horizons H in the MDP vs horizons H+ in the BAMDP,
        - resetting the tasks
        - adding the done info to the state (might be needed to make states markov)
        """
        super().__init__(env)
        
        # make sure we can call these attributes even if the orig env does not have them
        if not hasattr(self.env.unwrapped, 'task_dim'):
            self.env.unwrapped.task_dim = 0
        if not hasattr(self.env.unwrapped, 'belief_dim'):
            self.env.unwrapped.belief_dim = 0
        if not hasattr(self.env.unwrapped, 'get_belief'):
            self.env.unwrapped.get_belief = lambda: None
        if not hasattr(self.env.unwrapped, 'num_states'):
            self.env.unwrapped.num_states = None
        
        if add_done_info is None:
            if episodes_per_task > 1:
                self.add_done_info = True
            else:
                self.add_done_info = False
        else:
            self.add_done_info = add_done_info
        
        if self.add_done_info:
            if isinstance(self.observation_space, spaces.Box):
                if len(self.observation_space.shape) > 1:
                    raise ValueError  # can't add additional info for obs of more than 1D
                low = np.concatenate(
                    (self.observation_space.low.astype(np.float32), np.array([0.0], dtype=np.float32))
                )
                high = np.concatenate(
                    (self.observation_space.high.astype(np.float32), np.array([1.0], dtype=np.float32))
                )
                # shape will be deduced from low/high arrays
                self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
            else:
                raise NotImplementedError
        
        # calculate horizon length H^+
        self.episodes_per_task = episodes_per_task
        # counts the number of episodes
        self.episode_count = 0
        
        # count timesteps in BAMDP
        self.step_count_bamdp = 0.0
        
        # this tells us if we have reached the horizon in the underlying MDP
        self.done_mdp = True
    
    def reset(self, task=None):
        """ Resets the BAMDP """
        # reset task (Gymnasium wrappers like OrderEnforcing may not expose custom methods)
        self.env.unwrapped.reset_task(task)
        # normal reset
        try:
            state = self.env.reset()
        except AttributeError:
            state = self.env.unwrapped.reset()
        if isinstance(state, tuple):
            state = state[0]
        
        self.episode_count = 0
        self.step_count_bamdp = 0
        self.done_mdp = False
        if self.add_done_info:
            state = np.concatenate((state, [0.0]))
        return state
    
    def reset_mdp(self):
        """ Resets the underlying MDP only (*not* the task). """
        state = self.env.reset()
        if isinstance(state, tuple):
            state = state[0]
        if self.add_done_info:
            state = np.concatenate((state, [0.0]))
        self.done_mdp = False
        return state
    
    def step(self, action):
        # do normal environment step in MDP
        state, reward, terminated, truncated, info = self.env.step(action)
        self.done_mdp = bool(terminated or truncated)
        
        info['done_mdp'] = self.done_mdp
        
        if self.add_done_info:
            state = np.concatenate((state, [float(self.done_mdp)]))
        
        self.step_count_bamdp += 1
        # if we want to maximise performance over multiple episodes,
        # only say "done" when we collected enough episodes in this task
        terminated_bamdp = False
        truncated_bamdp = False
        if self.done_mdp:
            self.episode_count += 1
            if self.episode_count == self.episodes_per_task:
                terminated_bamdp = True
        
        if self.done_mdp and not terminated_bamdp:
            info['start_state'] = self.reset_mdp()
        return state, reward, terminated_bamdp, truncated_bamdp, info
