import argparse

import torch

from utils.cli import boolean_argument


def get_args(rest_args=None):
    parser = argparse.ArgumentParser(conflict_handler='resolve')

    # --- GENERAL ---

    # train parameters
    parser.add_argument('--num_frames', type=int, default=25e6,
                        help='number of frames to train on (excluding validation).')

    # test parameters
    parser.add_argument('--test', type=boolean_argument, default=False,
                        help='whether to test the environment on the train levels and held-out levels of same size sequentially.')

    # environment: --env_name assigned in main.py
    parser.add_argument('--distribution_mode', type=str, default='easy',
                        help='What variant of the levels to use {easy, hard, extreme, memory, exploration}.\
                              All games support `easy` and `hard`, while other options are game-specific.\
                              The default is `hard`. Switching to `easy` will reduce the number of timesteps\
                              required to solve each game and is useful for testing or when working with limited compute resources.')
    parser.add_argument('--paint_vel_info', type=boolean_argument, default=False,
                        help='paint player velocity info in the top left corner. Only supported by certain games.')
    parser.add_argument('--train_num_levels', type=int, default=200,
                        help='the number of unique levels that can be generated. Set to 0 to use unlimited levels.')
    parser.add_argument('--train_start_level', type=int, default=0,
                        help='the point in the list of levels available to the environment at which to index into.\
                              eg. --num_levels 50 --start_level 50 makes levels 50-99 available to this environment.')
    parser.add_argument('--test_num_levels', type=int, default=1000,
                        help='the number of unique levels that can be generated. Set to 0 to use unlimited levels.')
    parser.add_argument('--test_start_level', type=int, default=500000,
                        help='the point in the list of levels available to the environment at which to index into.\
                              eg. --num_levels 50 --start_level 50 makes levels 50-99 available to this environment.')

    # general settings
    parser.add_argument('--seed', type=int, default=10,
                        help='random seed (default: 10).')
    parser.add_argument('--deterministic_execution', type=boolean_argument, default=False,
                        help='Make code fully deterministic. Expects 1 process and uses deterministic CUDNN.')
    parser.add_argument('--num_processes', type=int, default=64,
                        help='how many training CPU processes to use (default: 64.')
    parser.add_argument('--num_frame_stack', type=int, default=0,
                        help='number of frames to stack for environments (default: 0).')

    # --- POLICY ---

    # algo
    parser.add_argument("--algo", type=str, default='ppo',
                        help='RL algorithm to use.')

    # network
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='number of nodes in hidden layer of the intermediate layer e.g. bottleneck or fully connected (default: 256).')

    # other hyperparameters
    parser.add_argument('--policy_gamma', type=float, default=0.999,
                        help='discount factor for rewards (default: 0.999).')
    parser.add_argument('--policy_gae_lambda', type=float, default=0.95,
                        help='factor for trade-off of bias vs variance for generalised advantage estimator (default: 0.95).')
    parser.add_argument('--policy_lr', type=float, default=5e-4,
                        help='learning rate (default: 5e-4).')
    parser.add_argument('--policy_num_steps', type=int, default=256,
                        help='number of steps of the vectorised env per update (i.e. batch size is n_steps * n_env.')
    parser.add_argument('--policy_ppo_epoch', type=int, default=3,
                        help='number of training epochs per update for ppo (default: 3).')
    parser.add_argument('--policy_num_mini_batch', type=int, default=8,
                        help='number of training minibatches per update for ppo For recurrent policies,\
                              should be smaller or equal than number of environments run in parallel (default: 8).')
    parser.add_argument('--policy_clip_param', type=float, default=0.2,
                        help='clip parameter for ppo (default: 0.2).')
    parser.add_argument('--policy_entropy_coef', type=float, default=0.01,
                        help='entropy term coefficient in the optimisation objective (default: 0.01).')
    parser.add_argument('--policy_value_loss_coef', type=float, default=0.5,
                        help='value function loss coefficient in the optimisation objective (default: 0.5).')
    parser.add_argument('--policy_max_grad_norm', type=float, default=0.5,
                        help='the maximum value for the gradient clipping (default: 0.5).')
    parser.add_argument('--policy_eps', type=float, default=1e-5,
                        help='Adam and RMSprop optimiser epsilon (default: 1e-5).')

    # --- OTHER ---

    # logging, saving and evaluation
    parser.add_argument('--log_interval', type=int, default=10,
                        help='number of timesteps between logging events (default: 10).')
    parser.add_argument('--save_interval', type=int, default=2000,
                        help='number of timesteps between saving events (default: 2000).')

    # Weights & Biases logging
    parser.add_argument("--proj_name", type=str, default='aup',
                        help="the name of the project to which this run will belong.")
    parser.add_argument("--run_name", type=str, default=None,
                        help="a display name for this run (default: env_name_time).")
    parser.add_argument("--group_name", type=str, default=None,
                        help="a string by which to group other runs.")

    if rest_args is not None:
        args = parser.parse_args(rest_args)
        args.cuda = torch.cuda.is_available()
        return args
    else:
        return parser
