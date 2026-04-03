"""
Main scripts to start experiments.
Takes a flag --env-type (see below for choices) and loads the parameters from the respective config file.
"""
import argparse
import warnings
import torch

# get configs
from config.gridworld import \
    args_grid_belief_oracle, args_grid_rl2, args_grid_varibad
from learner import Learner
from metalearner import MetaLearner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-type', default='gridworld_varibad')
    args, rest_args = parser.parse_known_args()
    env = args.env_type
    
    if env == 'gridworld_belief_oracle':
        args = args_grid_belief_oracle.get_args(rest_args)
    elif env == 'gridworld_varibad':
        args = args_grid_varibad.get_args(rest_args)
    else:
        raise Exception("Invalid Environment")

    # warning for deterministic execution
    if args.deterministic_execution:
        print('Envoking deterministic code execution.')
        if torch.backends.cudnn.enabled:
            warnings.warn('Running with deterministic CUDNN.')
        if args.num_processes > 1:
            raise RuntimeError('If you want fully deterministic code, run it with num_processes=1.'
                               'Warning: This will slow things down and might break A2C if '
                               'policy_num_steps < env._max_episode_steps.')

    # clean up arguments
    if args.disable_metalearner or args.disable_decoder:
        args.decode_reward = False
        args.decode_state = False
        args.decode_task = False

    # begin training (loop through all passed seeds)
    seed_list = [args.seed] if isinstance(args.seed, int) else args.seed
    for seed in seed_list:
        print('training', seed)
        args.seed = seed
        args.action_space = None

        if args.disable_metalearner:
            # If `disable_metalearner` is true, the file `learner.py` will be used instead of `metalearner.py`.
            # This is a stripped down version without encoder, decoder, stochastic latent variables, etc.
            learner = Learner(args)
        else:
            learner = MetaLearner(args)
        learner.train()

if __name__ == '__main__':
    main()
