import tensorflow as tf
import util


class ReplayInputs(object):
  def __init__(self, t, config):
    self.t = t

    with tf.name_scope(util.format_offset('replay', t)):
      shape = [None, config.input_frames] + list(config.input_shape)
      frames = tf.placeholder(tf.uint8, shape, 'frames')
      frames.feed_data = self.batch_frames
      self.frames = tf.to_float(frames) / 255.0

      # Inputs are expanded to shape [None, 1] to allow broadcasting
      #   and avoid operations creating shapes like [None, None, ...]
      action = tf.placeholder(tf.int32, [None], name='action')
      action.feed_data = self.batch_actions
      self.action = tf.expand_dims(action, axis=1, name='action')

      reward = tf.placeholder(tf.float32, [None], name='reward')
      reward.feed_data = self.batch_rewards
      self.reward = tf.expand_dims(reward, axis=1, name='reward')

      alive = tf.placeholder(tf.float32, [None], name='alive')
      alive.feed_data = self.batch_alives
      self.alive = tf.expand_dims(alive, axis=1, name='alive')

      total_reward = tf.placeholder(tf.float32, [None], name='total_reward')
      total_reward.feed_data = self.batch_total_rewards
      self.total_reward = tf.expand_dims(
          total_reward, axis=1, name='total_reward')

  def batch_frames(self, batch):
    return batch.frames(self.t)

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
    with tf.name_scope('global'):
      self.bootstrap_mask = tf.placeholder(
          tf.float32, [None, config.num_bootstrap_heads],
          name='bootstrap_mask')
      self.bootstrap_mask.feed_data = self.batch_bootstrap_mask

      importance_sampling = tf.placeholder(
          tf.float32, [None], name='importance_sampling')
      importance_sampling.feed_data = self.batch_importance_sampling
      self.importance_sampling = tf.expand_dims(
          importance_sampling, axis=1, name='importance_sampling')

  def batch_bootstrap_mask(self, batch):
    return batch.bootstrap_mask()

  def batch_importance_sampling(self, batch):
    return batch.importance_sampling()
