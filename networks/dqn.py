import math
import numpy as np
import tensorflow as tf

import util


class QNetwork(object):
  def __init__(self, scope, inputs, reward_scaling, config, reuse):
    self.scope = scope
    self.inputs = inputs
    self.config = config
    self.num_heads = config.num_bootstrap_heads
    self.using_ensemble = config.bootstrap_use_ensemble

    with tf.variable_scope(scope, reuse=reuse):
      conv_output = self.build_conv_layers(inputs)

      if config.actor_critic:
        self.build_actor_critic_heads(inputs, conv_output, reward_scaling)
      else:
        self.build_action_value_heads(inputs, conv_output, reward_scaling)

      if self.using_ensemble:
        self.build_ensemble()

    self.sample_head()

  def build_conv_layers(self, inputs):
    nhwc = tf.transpose(inputs.frames, [0, 2, 3, 1])
    conv1 = tf.layers.conv2d(
        nhwc, filters=32, kernel_size=[8, 8], strides=[4, 4], name='conv1')
    conv2 = tf.layers.conv2d(
        conv1, filters=64, kernel_size=[4, 4], strides=[2, 2], name='conv2')
    conv3 = tf.layers.conv2d(
        conv2, filters=64, kernel_size=[3, 3], strides=[1, 1], name='conv3')
    conv_output = tf.reshape(conv3, [-1, 64 * 7 * 7])

    # Rescale gradients entering the last convolution layer
    dueling_scale = 1.0 / math.sqrt(2) if self.config.dueling else 1.0
    scale = dueling_scale / self.num_heads
    if scale < 1:
      conv_output = util.scale_gradient(conv_output, scale)

    return conv_output

  def build_action_value_heads(self, inputs, conv_output, reward_scaling):
    self.heads = [
        ActionValueHead('head-%d' % i, inputs, conv_output, reward_scaling,
                        self.config) for i in range(self.num_heads)
    ]

    self.action_values = tf.stack(
        [head.action_values for head in self.heads],
        axis=1,
        name='action_values')

    self.taken_action_value = self.action_value(
        inputs.action, name='taken_action_value')

    value, greedy_action = tf.nn.top_k(self.action_values, k=1)
    self.value = tf.squeeze(value, axis=2, name='value')
    self.greedy_action = tf.squeeze(
        greedy_action, axis=2, name='greedy_action')

  def action_value(self, action, name='action_value'):
    return self.choose_from_actions(self.action_values, action, name)

  def build_actor_critic_heads(self, inputs, conv_output, reward_scaling):
    self.heads = [
        ActorCriticHead('head-%d' % i, inputs, conv_output, reward_scaling,
                        self.config) for i in range(self.num_heads)
    ]

    self.value = tf.stack(
        [head.value for head in self.heads], axis=1, name='value')
    self.greedy_action = tf.stack(
        [head.greedy_action for head in self.heads],
        axis=1,
        name='greedy_action')

    self.policy = tf.stack(
        [head.policy for head in self.heads], axis=1, name='policy')
    self._log_policy = tf.stack(
        [head.log_policy for head in self.heads], axis=1, name='log_policy')
    self.entropy = tf.reduce_sum(
        -self.policy * self._log_policy, axis=2, name='entropy')

  def log_policy(self, action, name='log_policy'):
    return self.choose_from_actions(self._log_policy, action, name)

  def choose_from_actions(self, actions, action, name):
    return tf.reduce_sum(
        actions * tf.one_hot(action, self.config.num_actions),
        axis=2,
        name=name)

  def build_ensemble(self):
    ensemble_votes = tf.reduce_sum(
        tf.one_hot(self.greedy_action, self.config.num_actions), axis=1)

    # Add some noise to break ties
    noise = tf.random_uniform([self.config.num_actions])

    _, ensemble_greedy_action = tf.nn.top_k(ensemble_votes + noise, k=1)
    self.ensemble_greedy_action = tf.squeeze(
        ensemble_greedy_action, axis=1, name='ensemble_greedy_action')

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
  def __init__(self, name, inputs, conv_outputs, reward_scaling, config):
    with tf.variable_scope(name):
      action_values = self.action_value_layer(conv_outputs, config)
      action_values = reward_scaling.unnormalize_output(action_values)
      value, greedy_action = tf.nn.top_k(action_values, k=1)

      self.action_values = tf.multiply(
          inputs.alive, action_values, name='action_values')
      self.value = tf.squeeze(inputs.alive * value, axis=1, name='value')
      self.greedy_action = tf.squeeze(
          greedy_action, axis=1, name='greedy_action')

  def action_value_layer(self, conv_outputs, config):
    if config.dueling:
      hidden_value = tf.layers.dense(
          conv_outputs, 256, tf.nn.relu, name='hidden_value')
      value = tf.layers.dense(hidden_value, 1, name='value')

      hidden_actions = tf.layers.dense(
          conv_outputs, 256, tf.nn.relu, name='hidden_actions')
      actions = tf.layers.dense(
          hidden_actions, config.num_actions, name='actions')

      return value + actions - tf.reduce_mean(actions, axis=1, keep_dims=True)

    else:
      hidden = tf.layers.dense(conv_outputs, 256, tf.nn.relu, name='hidden')
      return tf.layers.dense(hidden, config.num_actions, name='action_value')


class ActorCriticHead(object):
  def __init__(self, name, inputs, conv_outputs, reward_scaling, config):
    with tf.variable_scope(name):
      hidden = tf.layers.dense(conv_outputs, 256, tf.nn.relu, name='hidden')

      value = tf.layers.dense(hidden, 1)
      self.value = tf.squeeze(
          inputs.alive * reward_scaling.unnormalize_output(value),
          axis=1,
          name='value')

      actions = tf.layers.dense(hidden, config.num_actions, name='actions')
      self.policy = tf.nn.softmax(actions, name='policy')
      self.log_policy = tf.nn.log_softmax(actions, name='log_policy')

      # Sample action from policy
      self.greedy_action = tf.squeeze(
          tf.multinomial(
              self.log_policy, num_samples=1),
          axis=1,
          name='greedy_action')


class PolicyNetwork(QNetwork):
  def __init__(self, inputs, reward_scaling, config, reuse):
    super(PolicyNetwork, self).__init__('policy', inputs, reward_scaling,
                                        config, reuse)


class TargetNetwork(QNetwork):
  def __init__(self, inputs, reward_scaling, config, reuse):
    super(TargetNetwork, self).__init__('target', inputs, reward_scaling,
                                        config, reuse)
