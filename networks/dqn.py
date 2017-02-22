import math
import numpy as np
import tensorflow as tf

import util


class QNetwork(object):
  def __init__(self, scope, inputs, reward_scaling, config, reuse):
    self.scope = scope
    self.inputs = inputs
    self.num_actions = config.num_actions
    self.num_heads = config.num_bootstrap_heads

    with tf.variable_scope(scope, reuse=reuse):
      self.conv_layers = self.build_conv_layers(config)
      self.build_heads(self.conv_layers, reward_scaling, config)
      if config.bootstrap_use_ensemble:
        self.build_ensemble(config)

    self.sample_head()

  def build_conv_layers(self, config):
    nhwc = tf.transpose(self.inputs.frames, [0, 2, 3, 1])
    conv1 = tf.layers.conv2d(
        nhwc, filters=32, kernel_size=[8, 8], strides=[4, 4], name='conv1')
    conv2 = tf.layers.conv2d(
        conv1, filters=64, kernel_size=[4, 4], strides=[2, 2], name='conv2')
    conv3 = tf.layers.conv2d(
        conv2, filters=64, kernel_size=[3, 3], strides=[1, 1], name='conv3')
    conv_output = tf.reshape(conv3, [-1, 64 * 7 * 7])

    # Rescale gradients entering the last convolution layer
    scale = (1.0 / math.sqrt(2) if config.dueling else 1.0) / self.num_heads
    if scale < 1:
      conv_output = util.scale_gradient(conv_output, scale)

    return conv_output

  def build_heads(self, conv_output, reward_scaling, config):
    if config.actor_critic:
      self.heads = [
          ActorCriticHead('head-%d' % i, conv_output, reward_scaling, config)
          for i in range(self.num_heads)
      ]

      self.value = tf.stack([head.value for head in self.heads], axis=1)
      self.greedy_action = tf.stack(
          [self.greedy_action for head in self.heads], axis=1)

    else:
      self.heads = [
          ActionValueHead('head-%d' % i, conv_output, reward_scaling, config)
          for i in range(self.num_heads)
      ]

      self.action_values = tf.stack(
          [head.action_values for head in self.heads], axis=1)

      action_input = tf.expand_dims(self.inputs.action, axis=1)
      self.taken_action_value = self.action_value(
          action_input, name='taken_action_values')

      value, greedy_action = tf.nn.top_k(self.action_values, k=1)
      self.value = tf.squeeze(value, axis=2, name='values')
      self.greedy_action = tf.squeeze(
          greedy_action, axis=2, name='greedy_actions')

  def build_ensemble(self, config):
    ensemble_votes = tf.reduce_sum(
        tf.one_hot(self.greedy_action, config.num_actions), axis=1)
    # Add some noise to break ties
    noise = tf.random_uniform([config.num_actions])
    _, ensemble_greedy_action = tf.nn.top_k(ensemble_votes + noise, k=1)
    self.ensemble_greedy_action = tf.squeeze(
        ensemble_greedy_action, axis=1, name='ensemble_greedy_action')

  def action_value(self, action, name=None):
    return tf.reduce_sum(
        self.action_values * tf.one_hot(action, self.num_actions),
        axis=2,
        name=name)

  def sample_head(self):
    self.active_head = self.heads[np.random.randint(self.num_heads)]

  @property
  def choose_action(self):
    if self.num_heads == 1 or not self.using_ensemble:
      return self.active_head.greedy_action
    else:
      return self.ensemble_greedy_action

  @property
  def variables(self):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

  def copy_to_network(self, to_network):
    """Copy the tensor variables values between the two scopes"""

    operations = []

    for from_var, to_var in zip(self.variables, to_network.variables):
      operations.append(to_var.assign(from_var).op)

    with tf.control_dependencies(operations):
      return tf.no_op(name='copy_q_network')


class ActionValueHead(object):
  def __init__(self, name, inputs, reward_scaling, config):
    with tf.variable_scope(name):
      action_values = self.action_value_layer(inputs, config)
      action_values = reward_scaling.unnormalize_output(action_values)
      self.action_values = tf.identity(action_values, name='action_values')

      value, greedy_action = tf.nn.top_k(self.action_values, k=1)
      self.value = tf.squeeze(value, axis=1, name='value')
      self.greedy_action = tf.squeeze(
          greedy_action, axis=1, name='greedy_action')

  def action_value_layer(self, inputs, config):
    if config.dueling:
      hidden_value = tf.layers.dense(
          inputs, 256, tf.nn.relu, name='hidden_value')
      value = tf.layers.dense(hidden_value, 1, name='value')

      hidden_actions = tf.layers.dense(
          inputs, 256, tf.nn.relu, name='hidden_actions')
      actions = tf.layers.dense(
          hidden_actions, config.num_actions, name='actions')

      return value + actions - tf.reduce_mean(actions, axis=1, keep_dims=True)

    else:
      hidden = tf.layers.dense(inputs, 256, tf.nn.relu, name='hidden')
      return tf.layers.dense(hidden, config.num_actions, name='action_value')


class ActorCriticHead(object):
  def __init__(self, name, inputs, reward_scaling, config):
    with tf.variable_scope(name):
      hidden = tf.layers.dense(inputs, 256, tf.nn.relu, name='hidden')

      value = tf.layers.dense(hidden, 1, name='value')
      self.value = reward_scaling.unnormalize_output(value)

      actions = tf.layers.dense(hidden, config.num_actions, name='actions')
      self.policy = tf.nn.softmax(actions, name='policy')

      # Sample action from policy
      self.greedy_action = tf.multinomial(
          tf.nn.log_softmax(actions), 0, name='greedy_action')


class PolicyNetwork(QNetwork):
  def __init__(self, inputs, reward_scaling, config, reuse):
    super(PolicyNetwork, self).__init__('policy', inputs, reward_scaling,
                                        config, reuse)


class TargetNetwork(QNetwork):
  def __init__(self, inputs, reward_scaling, config, reuse):
    super(TargetNetwork, self).__init__('target', inputs, reward_scaling,
                                        config, reuse)
