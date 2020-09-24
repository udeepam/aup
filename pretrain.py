import argparse
import time
import datetime
from collections import defaultdict, deque

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

import wandb

# get configs
from configs import args_pretrain_aup

from utils.make_envs import make_vec_envs
from utils import helpers as utl
from utils import math as utl_math

from models.cb_vae import CBVAE, cb_vae_loss
from models.q_aux_model import QModel
from algorithms.ppo import PPO
from algorithms.storage import RolloutStorage


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# environment variable sets the number of threads to use for parallel regions
# https://github.com/ray-project/ray/issues/6962
# os.environ["OMP_NUM_THREADS"] = "1"


def main():

    # --- ARGUMENTS ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='coinrun',
                        help='name of the environment to train on.')
    parser.add_argument('--model', type=str, default='ppo',
                        help='the model to use for training.')
    args, rest_args = parser.parse_known_args()
    env_name = args.env_name
    model = args.model

    # get arguments
    args = args_pretrain_aup.get_args(rest_args)
    # place other args back into argparse.Namespace
    args.env_name = env_name
    args.model = model

    # Weights & Biases logger
    if args.run_name is None:
        # make run name as {env_name}_{TIME}
        now = datetime.datetime.now().strftime('_%d-%m_%H:%M:%S')
        args.run_name = args.env_name+'_'+args.algo+now
    # initialise wandb
    wandb.init(name=args.run_name,
               project=args.proj_name,
               group=args.group_name,
               config=args,
               monitor_gym=False)
    # save wandb dir path
    args.run_dir = wandb.run.dir
    wandb.config.update(args)

    # set random seed of random, torch and numpy
    utl.set_global_seed(args.seed, args.deterministic_execution)

    # --- OBTAIN DATA FOR TRAINING R_aux ---
    print("Gathering data for R_aux Model.")

    # gather observations for pretraining the auxiliary reward function (CB-VAE)
    envs = make_vec_envs(env_name=args.env_name,
                         start_level=0,
                         num_levels=0,
                         distribution_mode=args.distribution_mode,
                         paint_vel_info=args.paint_vel_info,
                         num_processes=args.num_processes,
                         num_frame_stack=args.num_frame_stack,
                         device=device)

    # number of frames ÷ number of policy steps before update ÷ number of cpu processes
    num_batch = args.num_processes * args.policy_num_steps
    num_updates = int(args.num_frames_r_aux) // num_batch

    # create list to store env observations
    obs_data = torch.zeros(num_updates*args.policy_num_steps + 1, args.num_processes, *envs.observation_space.shape)
    # reset environments
    obs = envs.reset()  # obs.shape = (n_env,C,H,W)
    obs_data[0].copy_(obs)
    obs = obs.to(device)

    for iter_idx in range(num_updates):
        # rollout policy to collect num_batch of experience and store in storage
        for step in range(args.policy_num_steps):
            # sample actions from random agent
            action = torch.randint(0, envs.action_space.n, (args.num_processes, 1))
            # observe rewards and next obs
            obs, reward, done, infos = envs.step(action)
            # store obs
            obs_data[1+iter_idx*args.policy_num_steps + step].copy_(obs)
    # close envs
    envs.close()

    # --- TRAIN R_aux (CB-VAE) ---
    # define CB-VAE where the encoder will be used as the auxiliary reward function R_aux
    print("Training R_aux Model.")

    # create dataloader for observations gathered
    obs_data = obs_data.reshape(-1, *envs.observation_space.shape)
    sampler = BatchSampler(SubsetRandomSampler(range(obs_data.size(0))),
                           args.cb_vae_batch_size,
                           drop_last=False)

    # initialise CB-VAE
    cb_vae = CBVAE(obs_shape=envs.observation_space.shape,
                   latent_dim=args.cb_vae_latent_dim).to(device)
    # optimiser
    optimiser = torch.optim.Adam(cb_vae.parameters(), lr=args.cb_vae_learning_rate)
    # put CB-VAE into train mode
    cb_vae.train()

    measures = defaultdict(list)
    for epoch in range(args.cb_vae_epochs):
        print("Epoch: ", epoch)
        start_time = time.time()
        batch_loss = 0
        for indices in sampler:
            obs = obs_data[indices].to(device)
            # zero accumulated gradients
            cb_vae.zero_grad()
            # forward pass through CB-VAE
            recon_batch, mu, log_var = cb_vae(obs)
            # calculate loss
            loss = cb_vae_loss(recon_batch, obs, mu, log_var)
            # backpropogation: calculating gradients
            loss.backward()
            # update parameters of generator
            optimiser.step()

            # save loss per mini-batch
            batch_loss += loss.item()*obs.size(0)
        # log losses per epoch
        wandb.log({'cb_vae/loss': batch_loss / obs_data.size(0),
                   'cb_vae/time_taken': time.time()-start_time,
                   'cb_vae/epoch': epoch})
    indices = np.random.randint(0, obs.size(0), args.cb_vae_num_samples**2)
    measures['true_images'].append(obs[indices].detach().cpu().numpy())
    measures['recon_images'].append(recon_batch[indices].detach().cpu().numpy())

    # plot ground truth images
    plt.rcParams.update({'font.size': 10})
    fig, axs = plt.subplots(args.cb_vae_num_samples, args.cb_vae_num_samples, figsize=(20, 20))
    for i, img in enumerate(measures['true_images'][0]):
        axs[i//args.cb_vae_num_samples][i %args.cb_vae_num_samples].imshow(img.transpose(1, 2, 0))
        axs[i//args.cb_vae_num_samples][i %args.cb_vae_num_samples].axis('off')
    wandb.log({"Ground Truth Images": wandb.Image(plt)})
    # plot reconstructed images
    fig, axs = plt.subplots(args.cb_vae_num_samples, args.cb_vae_num_samples, figsize=(20, 20))
    for i, img in enumerate(measures['recon_images'][0]):
        axs[i//args.cb_vae_num_samples][i %args.cb_vae_num_samples].imshow(img.transpose(1, 2, 0))
        axs[i//args.cb_vae_num_samples][i %args.cb_vae_num_samples].axis('off')
    wandb.log({"Reconstructed Images": wandb.Image(plt)})

    # --- TRAIN Q_aux --
    # train PPO agent with value head replaced with action-value head and training on R_aux instead of the environment R
    print("Training Q_aux Model.")

    # initialise environments for training Q_aux
    envs = make_vec_envs(env_name=args.env_name,
                         start_level=0,
                         num_levels=0,
                         distribution_mode=args.distribution_mode,
                         paint_vel_info=args.paint_vel_info,
                         num_processes=args.num_processes,
                         num_frame_stack=args.num_frame_stack,
                         device=device)

    # initialise policy network
    actor_critic = QModel(obs_shape=envs.observation_space.shape,
                          action_space=envs.action_space,
                          hidden_size=args.hidden_size).to(device)

    # initialise policy trainer
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

    # initialise rollout storage for the policy
    rollouts = RolloutStorage(num_steps=args.policy_num_steps,
                              num_processes=args.num_processes,
                              obs_shape=envs.observation_space.shape,
                              action_space=envs.action_space)

    # count number of frames and updates
    frames   = 0
    iter_idx = 0

    update_start_time = time.time()
    # reset environments
    obs = envs.reset()  # obs.shape = (n_envs,C,H,W)
    # insert initial observation to rollout storage
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    # initialise buffer for calculating mean episodic returns
    episode_info_buf = deque(maxlen=10)

    # calculate number of updates
    # number of frames ÷ number of policy steps before update ÷ number of cpu processes
    args.num_batch = args.num_processes * args.policy_num_steps
    args.num_updates = int(args.num_frames_q_aux) // args.num_batch
    print("Number of updates: ", args.num_updates)
    for iter_idx in range(args.num_updates):
        print("Iter: ", iter_idx)

        # put actor-critic into train mode
        actor_critic.train()

        # rollout policy to collect num_batch of experience and store in storage
        for step in range(args.policy_num_steps):

            with torch.no_grad():
                # sample actions from policy
                value, action, action_log_prob = actor_critic.act(rollouts.obs[step])
                # obtain reward R_aux from encoder of CB-VAE
                r_aux, _, _ = cb_vae.encode(rollouts.obs[step])

            # observe rewards and next obs
            obs, _, done, infos = envs.step(action)

            # log episode info if episode finished
            for i, info in enumerate(infos):
                if 'episode' in info.keys():
                    episode_info_buf.append(info['episode'])
            # create mask for episode ends
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done]).to(device)

            # add experience to policy buffer
            rollouts.insert(obs,
                            r_aux,
                            action,
                            value,
                            action_log_prob,
                            masks)

            frames += args.num_processes

        # --- UPDATE ---

        # bootstrap next value prediction
        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1]).detach()

        # compute returns for current rollouts
        rollouts.compute_returns(next_value,
                                 args.policy_gamma,
                                 args.policy_gae_lambda)

        # update actor-critic using policy gradient algo
        total_loss, value_loss, action_loss, dist_entropy = policy.update(rollouts)

        # clean up after update
        rollouts.after_update()

        # --- LOGGING ---

        if iter_idx % args.log_interval == 0 or iter_idx == args.num_updates - 1:
            # get stats for run
            update_end_time = time.time()
            num_interval_updates = 1 if iter_idx == 0 else args.log_interval
            fps = num_interval_updates * (args.num_processes * args.policy_num_steps) / (update_end_time - update_start_time)
            update_start_time = update_end_time
            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            ev = utl_math.explained_variance(utl.sf01(rollouts.value_preds),
                                             utl.sf01(rollouts.returns))

            wandb.log({'q_aux_misc/timesteps': frames,
                       'q_aux_misc/fps': fps,
                       'q_aux_misc/explained_variance': float(ev),
                       'q_aux_losses/total_loss': total_loss,
                       'q_aux_losses/value_loss': value_loss,
                       'q_aux_losses/action_loss': action_loss,
                       'q_aux_losses/dist_entropy': dist_entropy,
                       'q_aux_train/mean_episodic_return': utl_math.safe_mean([episode_info['r'] for episode_info in episode_info_buf]),
                       'q_aux_train/mean_episodic_length': utl_math.safe_mean([episode_info['l'] for episode_info in episode_info_buf])})

    # close envs
    envs.close()

    # --- SAVE MODEL ---
    print("Saving Q_aux Model.")
    torch.save(actor_critic.state_dict(), args.q_aux_path)


if __name__ == '__main__':
    main()
