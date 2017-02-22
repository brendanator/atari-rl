import tensorflow as tf


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
      self.scale_weight = tf.get_variable('scale_weight', 1)
      self.scale_bias = tf.get_variable('scale_bias', 1)

      self.sigma_squared_input = tf.placeholder(tf.float32, (),
                                                'sigma_squared_input')
      self.sigma_squared_input.feed_data = self.batch_sigma_squared

  def batch_sigma_squared(self, batch):
    batch_size = len(batch)
    rewards = batch.rewards(0)

    average_reward = rewards.sum() / batch_size
    self.mu = (1 - self.beta) * self.mu + self.beta * average_reward

    average_square_reward = (rewards**2).sum() / batch_size
    self.v = (1 - self.beta) * self.v + self.beta * average_square_reward

    sigma_squared = (self.v - self.mu**2) / self.variance
    if sigma_squared > 0:
      return sigma_squared
    else:
      return 1.0

  def unnormalize_output(self, output):
    return output * self.scale_weight + self.scale_bias

  @property
  def variables(self):
    return [self.scale_weight, self.scale_bias]

  def scale_gradients(self, grads, variables_to_scale):
    grads_ = []
    for grad, var in grads:
      if grad is not None:
        if var in variables_to_scale:
          grad /= self.sigma_squared_input
        grads_.append((grad, var))

    return grads


class DisabledRewardScaling(object):
  """An implementation that doesn't scale rewards"""

  def __init__(self):
    pass

  def unnormalize_output(self, output):
    return output

  @property
  def variables(self):
    return []

  def scale_gradients(self, grads, variables_to_scale):
    return grads
