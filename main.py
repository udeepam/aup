"""
Based on: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
"""
import os
import argparse
import warnings
import random
import time
import datetime
from collections import deque, defaultdict
import math

import torch

import wandb

from configs import args_ppo, args_ppo_aup

from models.policy import ACModel
from algorithms.ppo import PPO
from algorithms.storage import RolloutStorage
from models.q_aux_model import QModel

from utils.make_envs import make_vec_envs
from utils import helpers as utl
from utils import math as utl_math
from utils import evaluation as utl_eval
from utils import test as utl_test


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# environment variable sets the number of threads to use for parallel regions
# https://github.com/ray-project/ray/issues/6962
# os.environ["OMP_NUM_THREADS"] = "1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='coinrun',
                        help='name of the environment to train on.')
    parser.add_argument('--model', type=str, default='ppo',
                        help='the model to use for training. {ppo, ppo_aup}')
    args, rest_args = parser.parse_known_args()
    env_name = args.env_name
    model = args.model

    # --- ARGUMENTS ---

    if model == 'ppo':
        args = args_ppo.get_args(rest_args)
    elif model == 'ppo_aup':
        args = args_ppo_aup.get_args(rest_args)
    else:
        raise NotImplementedError

    # place other args back into argparse.Namespace
    args.env_name = env_name
    args.model = model

    # warnings
    if args.deterministic_execution:
        print('Envoking deterministic code execution.')
        if torch.backends.cudnn.enabled:
            warnings.warn('Running with deterministic CUDNN.')
        if args.num_processes > 1:
            raise RuntimeError('If you want fully deterministic code, run it with num_processes=1.'
                               'Warning: This will slow things down and might break A2C if '
                               'policy_num_steps < env._max_episode_steps.')

    # --- TRAINING ---
    print("Setting up wandb logging.")

    # Weights & Biases logger
    if args.run_name is None:
        # make run name as {env_name}_{TIME}
        now = datetime.datetime.now().strftime('_%d-%m_%H:%M:%S')
        args.run_name = args.env_name+'_'+args.algo+now
    # initialise wandb
    wandb.init(project=args.proj_name,
               name=args.run_name,
               group=args.group_name,
               config=args,
               monitor_gym=False)
    # save wandb dir path
    args.run_dir = wandb.run.dir
    # make directory for saving models
    save_dir = os.path.join(wandb.run.dir, 'models')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # set random seed of random, torch and numpy
    utl.set_global_seed(args.seed, args.deterministic_execution)

    print("Setting up Environments.")
    # initialise environments for training
    train_envs = make_vec_envs(env_name=args.env_name,
                               start_level=args.train_start_level,
                               num_levels=args.train_num_levels,
                               distribution_mode=args.distribution_mode,
                               paint_vel_info=args.paint_vel_info,
                               num_processes=args.num_processes,
                               num_frame_stack=args.num_frame_stack,
                               device=device)
    # initialise environments for evaluation
    eval_envs = make_vec_envs(env_name=args.env_name,
                              start_level=0,
                              num_levels=0,
                              distribution_mode=args.distribution_mode,
                              paint_vel_info=args.paint_vel_info,
                              num_processes=args.num_processes,
                              num_frame_stack=args.num_frame_stack,
                              device=device)
    _ = eval_envs.reset()

    print("Setting up Actor-Critic model and Training algorithm.")
    # initialise policy network
    actor_critic = ACModel(obs_shape=train_envs.observation_space.shape,
                           action_space=train_envs.action_space,
                           hidden_size=args.hidden_size).to(device)

    # initialise policy training algorithm
    if args.algo == 'ppo':
        policy = PPO(actor_critic=actor_critic,
                     ppo_epoch=args.policy_ppo_epoch,
                     num_mini_batch=args.policy_num_mini_batch,
                     clip_param=args.policy_clip_param,
                     value_loss_coef=args.policy_value_loss_coef,
                     entropy_coef=args.policy_entropy_coef,
                     max_grad_norm=args.policy_max_grad_norm,
                     lr=args.policy_lr,
                     eps=args.policy_eps)
    else:
        raise NotImplementedError

    # initialise rollout storage for the policy training algorithm
    rollouts = RolloutStorage(num_steps=args.policy_num_steps,
                              num_processes=args.num_processes,
                              obs_shape=train_envs.observation_space.shape,
                              action_space=train_envs.action_space)

    # initialise Q_aux function(s) for AUP
    if args.use_aup:
        print("Initialising Q_aux models.")
        q_aux = [QModel(obs_shape=train_envs.observation_space.shape,
                        action_space=train_envs.action_space,
                        hidden_size=args.hidden_size).to(device) for _ in range(args.num_q_aux)]
        if args.num_q_aux==1:
            # load weights to model
            path = args.q_aux_dir+"0.pt"
            q_aux[0].load_state_dict(torch.load(path))
            q_aux[0].eval()
        else:
            # get max number of q_aux functions to choose from
            args.max_num_q_aux = os.listdir(args.q_aux_dir)
            q_aux_models = random.sample(list(range(0, args.max_num_q_aux)), args.num_q_aux)
            # load weights to models
            for i, model in enumerate(q_aux):
                path = args.q_aux_dir+str(q_aux_models[i])+".pt"
                model.load_state_dict(torch.load(path))
                model.eval()

    # count number of frames and updates
    frames = 0
    iter_idx = 0

    # update wandb args
    wandb.config.update(args)

    update_start_time = time.time()
    # reset environments
    obs = train_envs.reset()  # obs.shape = (num_processes,C,H,W)
    # insert initial observation to rollout storage
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    # initialise buffer for calculating mean episodic returns
    episode_info_buf = deque(maxlen=10)

    # calculate number of updates
    # number of frames ÷ number of policy steps before update ÷ number of processes
    args.num_batch = args.num_processes * args.policy_num_steps
    args.num_updates = int(args.num_frames) // args.num_batch

    # define AUP coefficient
    if args.use_aup:
        aup_coef = args.aup_coef_start
        aup_linear_increase_val = math.exp(math.log(args.aup_coef_end/args.aup_coef_start)/args.num_updates)

    print("Training beginning.")
    print("Number of updates: ", args.num_updates)
    for iter_idx in range(args.num_updates):
        print("Iter: ", iter_idx)

        # put actor-critic into train mode
        actor_critic.train()

        if args.use_aup:
            aup_measures = defaultdict(list)

        # rollout policy to collect num_batch of experience and place in storage
        for step in range(args.policy_num_steps):

            # sample actions from policy
            with torch.no_grad():
                value, action, action_log_prob = actor_critic.act(rollouts.obs[step])

            # observe rewards and next obs
            obs, reward, done, infos = train_envs.step(action)

            # calculate AUP reward
            if args.use_aup:
                intrinsic_reward = torch.zeros_like(reward)
                with torch.no_grad():
                    for model in q_aux:
                        # get action-values
                        action_values = model.get_action_value(rollouts.obs[step])
                        # get action-value for action taken
                        action_value = torch.sum(action_values*torch.nn.functional.one_hot(action, num_classes=train_envs.action_space.n).squeeze(dim=1), dim=1)
                        # calculate the penalty
                        intrinsic_reward += torch.abs(action_value.unsqueeze(dim=1) - action_values[:, 4].unsqueeze(dim=1))
                intrinsic_reward /= args.num_q_aux
                # add intrinsic reward to the extrinsic reward
                reward -= aup_coef*intrinsic_reward
                # log the intrinsic reward from the first env.
                aup_measures['intrinsic_reward'].append(aup_coef*intrinsic_reward[0, 0])
                if done[0] and infos[0]['prev_level_complete']==1:
                    aup_measures['episode_complete'].append(2)
                elif done[0] and infos[0]['prev_level_complete']==0:
                    aup_measures['episode_complete'].append(1)
                else:
                    aup_measures['episode_complete'].append(0)

            # log episode info if episode finished
            for info in infos:
                if 'episode' in info.keys():
                    episode_info_buf.append(info['episode'])

            # create mask for episode ends
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done]).to(device)

            # add experience to storage
            rollouts.insert(obs,
                            reward,
                            action,
                            value,
                            action_log_prob,
                            masks)

            frames += args.num_processes

        # linearly increase aup coefficient after every update
        if args.use_aup:
            aup_coef *= aup_linear_increase_val

        # --- UPDATE ---

        # bootstrap next value prediction
        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1]).detach()

        # compute returns for current rollouts
        rollouts.compute_returns(next_value,
                                 args.policy_gamma,
                                 args.policy_gae_lambda)

        # update actor-critic using policy training algorithm
        total_loss, value_loss, action_loss, dist_entropy = policy.update(rollouts)

        # clean up storage after update
        rollouts.after_update()

        # --- LOGGING ---

        if iter_idx % args.log_interval == 0 or iter_idx == args.num_updates - 1:

            # --- EVALUATION ---
            eval_episode_info_buf = utl_eval.evaluate(eval_envs=eval_envs,
                                                      actor_critic=actor_critic,
                                                      device=device)

            # get stats for run
            update_end_time = time.time()
            num_interval_updates = 1 if iter_idx == 0 else args.log_interval
            fps = num_interval_updates * (args.num_processes * args.policy_num_steps) / (update_end_time - update_start_time)
            update_start_time = update_end_time
            # calculates whether the value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            ev = utl_math.explained_variance(utl.sf01(rollouts.value_preds),
                                             utl.sf01(rollouts.returns))

            if args.use_aup:
                step = frames - args.num_processes*args.policy_num_steps
                for i in range(args.policy_num_steps):
                    wandb.log({'aup/intrinsic_reward': aup_measures['intrinsic_reward'][i],
                               'aup/episode_complete': aup_measures['episode_complete'][i]}, step=step)
                    step += args.num_processes

            wandb.log({'misc/timesteps': frames,
                       'misc/fps': fps,
                       'misc/explained_variance': float(ev),
                       'losses/total_loss': total_loss,
                       'losses/value_loss': value_loss,
                       'losses/action_loss': action_loss,
                       'losses/dist_entropy': dist_entropy,
                       'train/mean_episodic_return': utl_math.safe_mean([episode_info['r'] for episode_info in episode_info_buf]),
                       'train/mean_episodic_length': utl_math.safe_mean([episode_info['l'] for episode_info in episode_info_buf]),
                       'eval/mean_episodic_return': utl_math.safe_mean([episode_info['r'] for episode_info in eval_episode_info_buf]),
                       'eval/mean_episodic_length': utl_math.safe_mean([episode_info['l'] for episode_info in eval_episode_info_buf])}, step=frames)

        # --- SAVE MODEL ---

        # save for every interval-th episode or for the last epoch
        if iter_idx !=0 and (iter_idx % args.save_interval == 0 or iter_idx == args.num_updates - 1):
            print("Saving Actor-Critic Model.")
            torch.save(actor_critic.state_dict(), os.path.join(save_dir, "policy{0}.pt".format(iter_idx)))

    # close envs
    train_envs.close()
    eval_envs.close()

    # --- TEST ---

    if args.test:
        print("Testing beginning.")
        episodic_return = utl_test.test(args=args,
                                        actor_critic=actor_critic,
                                        device=device)

        # save returns from train and test levels to analyse using interactive mode
        train_levels = torch.arange(args.train_start_level, args.train_start_level+args.train_num_levels)
        for i, level in enumerate(train_levels):
            wandb.log({'test/train_levels': level,
                       'test/train_returns': episodic_return[0][i]})
        test_levels = torch.arange(args.test_start_level, args.test_start_level+args.test_num_levels)
        for i, level in enumerate(test_levels):
            wandb.log({'test/test_levels': level,
                       'test/test_returns': episodic_return[1][i]})
        # log returns from test envs
        wandb.run.summary["train_mean_episodic_return"] = utl_math.safe_mean(episodic_return[0])
        wandb.run.summary["test_mean_episodic_return"]  = utl_math.safe_mean(episodic_return[1])


if __name__ == '__main__':
    main()
