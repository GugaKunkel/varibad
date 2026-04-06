"""
Main script to start experiments.
Loads gridworld_varibad parameters and starts training.
"""
from config import args_grid_varibad
from metalearner import MetaLearner

def main():
    args = args_grid_varibad.get_args(None)

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
