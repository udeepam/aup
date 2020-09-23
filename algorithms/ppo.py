"""
Taken from: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr
"""
import torch
import torch.nn as nn
import torch.optim as optim


class PPO():
    def __init__(self,
                 actor_critic,
                 ppo_epoch,
                 num_mini_batch,
                 clip_param,
                 value_loss_coef,
                 entropy_coef,
                 max_grad_norm,
                 lr,
                 eps):

        self.actor_critic = actor_critic

        # ppo parameters
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.clip_param = clip_param
        self.max_grad_norm = max_grad_norm

        # optimiser
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

    def update(self, rollouts):
        """
        Update model using PPO.
        """
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        # initialise epoch values
        total_loss_epoch = 0
        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        # iterate through experience stored in rollouts
        for _ in range(self.ppo_epoch):

            # get generator which batches experience stored in rollouts
            data_generator = rollouts.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, actions_batch, value_preds_batch, return_batch, old_action_log_probs_batch, adv_targ = sample

                # --- PPO ---
                values, action_log_probs, dist_entropy = self.actor_critic.evaluate_actions(obs_batch, actions_batch)

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                # get value loss
                value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - return_batch).pow(2)
                value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

                # zero accumulated gradients
                self.optimizer.zero_grad()
                # calculate loss
                loss = action_loss - dist_entropy * self.entropy_coef + value_loss * self.value_loss_coef
                # backpropogate: calculate gradients
                loss.backward()
                # clippling
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                # update parameters of model
                self.optimizer.step()

                # update epoch values
                total_loss_epoch += loss.item()
                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        # calculate losses for epoch
        total_loss_epoch /= num_updates
        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return total_loss_epoch, value_loss_epoch, action_loss_epoch, dist_entropy_epoch
