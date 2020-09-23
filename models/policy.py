"""
Based on: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr
"""
import torch.nn as nn

from models.impala_cnn import ImpalaCNN

from utils import helpers as utl
from utils import distributions as utl_dist


init_ = lambda m: utl.init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
init_actor_ = lambda m: utl.init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01)
init_relu_ = lambda m: utl.init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))


class ACModel(nn.Module):
    def __init__(self,
                 obs_shape,
                 action_space,
                 hidden_size=256):
        """
        Actor-critic network.
        """
        super(ACModel, self).__init__()

        self.num_actions = action_space.n

        # define feature extractor
        self.feature_extractor = ImpalaCNN(num_inputs=obs_shape[0], hidden_size=hidden_size)

        # define critic model
        self.critic_linear = init_(nn.Linear(hidden_size, 1))
        # define actor model
        self.actor_linear = init_actor_(nn.Linear(hidden_size, self.num_actions))

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
        value = self.critic_linear(x)
        # create action distribution
        dist = self.dist(logits=actor_features)
        # sample actions
        action = dist.sample()
        # get action log probabilities from distribution
        action_log_probs = dist.log_probs(action)
        return value, action, action_log_probs

    def get_value(self, inputs):
        x = self.feature_extractor(inputs)
        value = self.critic_linear(x)
        return value

    def evaluate_actions(self, inputs, action):
        x = self.feature_extractor(inputs)
        actor_features = self.actor_linear(x)
        value = self.critic_linear(x)
        # create action distribution
        dist = self.dist(logits=actor_features)
        # get action log probabilities from distribution
        action_log_probs = dist.log_probs(action)
        # calculate entropy
        dist_entropy = dist.entropy().mean()
        return value, action_log_probs, dist_entropy
