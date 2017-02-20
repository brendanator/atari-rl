import tensorflow as tf


class Inputs(object):
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

  def batch_frames(self, batch):
    return batch.observations(self.t)

  def batch_actions(self, batch):
    return batch.actions(self.t)

  def batch_rewards(self, batch):
    return batch.rewards(self.t)

  def batch_alives(self, batch):
    return batch.alives(self.t)
