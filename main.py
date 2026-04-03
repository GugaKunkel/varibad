"""
Main scripts to start experiments.
Loads gridworld_varibad parameters and starts training.
"""
import warnings
import torch

from config.gridworld import args_grid_varibad
from metalearner import MetaLearner


def main():
    args = args_grid_varibad.get_args(None)

    # warning for deterministic execution
    if args.deterministic_execution:
        print('Envoking deterministic code execution.')
        if torch.backends.cudnn.enabled:
            warnings.warn('Running with deterministic CUDNN.')
        if args.num_processes > 1:
            raise RuntimeError('If you want fully deterministic code, run it with num_processes=1.'
                               'Warning: This will slow things down and might break A2C if '
                               'policy_num_steps < _max_episode_steps.')

    # begin training (loop through all passed seeds)
    seed_list = [args.seed] if isinstance(args.seed, int) else args.seed
    for seed in seed_list:
        print('training', seed)
        args.seed = seed
        args.action_space = None

        learner = MetaLearner(args)
        learner.train()

if __name__ == '__main__':
    main()
