"""
Based on: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr
"""
import torch
import torch.nn as nn

from models.impala_cnn import ImpalaCNN

from utils import helpers as utl
from utils import distributions as utl_dist


init_ = lambda m: utl.init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
init_actor_ = lambda m: utl.init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01)
init_relu_ = lambda m: utl.init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))


class QModel(nn.Module):
    def __init__(self,
                 obs_shape,
                 action_space,
                 hidden_size=256):
        """
        Actor-critic network.
        """
        super(QModel, self).__init__()

        self.n_actions = action_space.n

        # define feature extractor
        self.feature_extractor = ImpalaCNN(num_inputs=obs_shape[0], hidden_size=hidden_size)

        # define action-value critic model: takes the state and outputs a vector of action-values for each action
        self.critic_linear = init_(nn.Linear(hidden_size, action_space.n))
        # define actor model
        self.actor_linear = init_actor_(nn.Linear(hidden_size, action_space.n))

        # intialise output distribution of the actor network
        if action_space.__class__.__name__ == "Discrete":
            self.dist = utl_dist.FixedCategorical
        else:
            raise NotImplementedError

        # put model into train mode
        self.train()

    def act(self, inputs):
        x = self.feature_extractor(inputs)
        actor_features = self.actor_linear(x)
        action_value = self.critic_linear(x)
        # create action distribution
        dist = self.dist(logits=actor_features)
        # sample actions
        action = dist.sample()
        # get action log probabilities from distribution
        action_log_probs = dist.log_probs(action)
        # calculate value as v(s) = E_{a~p(a|s)}[q(s,a)] = sum_a p(a|s)q(s,a)
        value = torch.sum(dist.probs*action_value, dim=1).unsqueeze(dim=1)
        return value, action, action_log_probs

    def get_value(self, inputs):
        x = self.feature_extractor(inputs)
        actor_features = self.actor_linear(x)
        action_value = self.critic_linear(x)
        # create action distribution
        dist = self.dist(logits=actor_features)
        # calculate value as v(s) = E_{a~p(a|s)}[q(s,a)] = sum_a p(a|s)q(s,a)
        value = torch.sum(dist.probs*action_value, dim=1).unsqueeze(dim=1)
        return value

    def evaluate_actions(self, inputs, action):
        x = self.feature_extractor(inputs)
        actor_features = self.actor_linear(x)
        action_value = self.critic_linear(x)
        # create action distribution
        dist = self.dist(logits=actor_features)
        # get action log probabilities from distribution
        action_log_probs = dist.log_probs(action)
        # calculate entropy
        dist_entropy = dist.entropy().mean()
        # calculate value as v(s) = E_{a~p(a|s)}[q(s,a)] = sum_a p(a|s)q(s,a)
        value = torch.sum(dist.probs*action_value, dim=1).unsqueeze(dim=1)
        return value, action_log_probs, dist_entropy

    def get_action_value(self, inputs):
        x = self.feature_extractor(inputs)
        action_value = self.critic_linear(x)
        return action_value
