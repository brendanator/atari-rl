import tensorflow as tf


class NetworkInputs(object):
  def __init__(self, t, config):
    self.t = t

    with tf.variable_scope('time_offset_%d' % t):
      shape = [None, config.input_frames] + config.input_shape
      self.input_frames = tf.placeholder(tf.float32, shape, 'input_frames')
      self.input_frames.feed_data = self.batch_frames

      self.action_input = tf.placeholder(tf.int32, [None], name='action')
      self.action_input.feed_data = self.batch_actions

      self.reward_input = tf.placeholder(tf.float32, [None], name='reward')
      self.reward_input.feed_data = self.batch_rewards

      self.alive_input = tf.placeholder(tf.float32, [None], name='alive')
      self.alive_input.feed_data = self.batch_alives

      self.total_reward_input = tf.placeholder(
          tf.float32, [None], name='total_reward')
      self.total_reward_input.feed_data = self.batch_total_rewards

  def batch_frames(self, batch):
    return batch.observations(self.t)

  def batch_actions(self, batch):
    return batch.actions(self.t)

  def batch_rewards(self, batch):
    return batch.rewards(self.t)

  def batch_alives(self, batch):
    return batch.alives(self.t)

  def batch_total_rewards(self, batch):
    return batch.total_rewards(self.t)


class GlobalInputs(object):
  def __init__(self, config):
    with tf.variable_scope('global_inputs'):
      self.bootstrap_mask = tf.placeholder(
          tf.float32, [None, config.num_bootstrap_heads],
          name='bootstrap_mask')
      self.bootstrap_mask.feed_data = self.batch_bootstrap_mask

      self.importance_sampling = tf.placeholder(tf.float32, [None],
                                                'importance_sampling')
      self.importance_sampling.feed_data = self.batch_importance_sampling

  def batch_bootstrap_mask(self, batch):
    return batch.bootstrap_mask()

  def batch_importance_sampling(self, batch):
    return batch.importance_sampling()
