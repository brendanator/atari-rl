import tensorflow as tf
import util


class RewardScaling(object):
  """
  Reward scaling is implemented using normalized SGD (algorithm 2 from paper)
  """

  def __init__(self, config):
    self.mu = 0
    self.v = 0
    self.beta = config.reward_scaling_beta
    self.variance = config.reward_scaling_stddev**2

    with tf.variable_scope('reward_scaling'):
      self.scale_weight = util.variable_with_weight_decay(
          'scale_weight', 1, wd=None)
      self.scale_bias = util.variable_with_weight_decay(
          'scale_bias', 1, wd=None)

      self.sigma_squared_input = tf.placeholder(tf.float32, (),
                                                'sigma_squared_input')

  @property
  def variables(self):
    return [self.scale_weight, self.scale_bias]

  def sigma_squared(self, reward_batch):
    batch_size = len(reward_batch)

    average_reward = reward_batch.sum() / batch_size
    self.mu = (1 - self.beta) * self.mu + self.beta * average_reward

    average_square_reward = (reward_batch**2).sum() / batch_size
    self.v = (1 - self.beta) * self.v + self.beta * average_square_reward

    sigma_squared = (self.v - self.mu**2) / self.variance
    if sigma_squared > 0:
      return sigma_squared
    else:
      return 1.0

  def unnormalize_output(self, output):
    return output * self.scale_weight + self.scale_bias
