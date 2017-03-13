import math
import numpy as np
import tensorflow as tf

import util


class Network(object):
  def __init__(self, variable_scope, inputs, reward_scaling, config,
               write_summaries):
    self.scope = variable_scope
    self.inputs = inputs
    self.config = config
    self.write_summaries = write_summaries
    self.num_heads = config.num_bootstrap_heads
    self.using_ensemble = config.bootstrap_use_ensemble

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
    self.activation_summary(nhwc)
    conv1 = tf.layers.conv2d(
        nhwc,
        filters=32,
        kernel_size=[8, 8],
        strides=[4, 4],
        activation=tf.nn.relu,
        name='conv1')
    self.activation_summary(conv1)
    conv2 = tf.layers.conv2d(
        conv1,
        filters=64,
        kernel_size=[4, 4],
        strides=[2, 2],
        activation=tf.nn.relu,
        name='conv2')
    self.activation_summary(conv2)
    conv3 = tf.layers.conv2d(
        conv2,
        filters=64,
        kernel_size=[3, 3],
        strides=[1, 1],
        activation=tf.nn.relu,
        name='conv3')
    self.activation_summary(conv3)
    conv_output = tf.reshape(conv3, [-1, 64 * 7 * 7])

    # Rescale gradients entering the last convolution layer
    dueling_scale = 1.0 / math.sqrt(2) if self.config.dueling else 1.0
    scale = dueling_scale / self.num_heads
    if scale < 1:
      conv_output = util.scale_gradient(conv_output, scale)

    return conv_output

  def build_action_value_heads(self, inputs, conv_output, reward_scaling):
    self.heads = [
        ActionValueHead('head%d' % i, inputs, conv_output, reward_scaling,
                        self.config) for i in range(self.num_heads)
    ]

    self.action_values = tf.stack(
        [head.action_values for head in self.heads],
        axis=1,
        name='action_values')
    self.activation_summary(self.action_values)

    self.taken_action_value = self.action_value(
        inputs.action, name='taken_action_value')

    value, greedy_action = tf.nn.top_k(self.action_values, k=1)
    self.value = tf.squeeze(value, axis=2, name='value')
    self.greedy_action = tf.squeeze(
        greedy_action, axis=2, name='greedy_action')

  def action_value(self, action, name='action_value'):
    with tf.name_scope(name):
      return self.choose_from_actions(self.action_values, action)

  def build_actor_critic_heads(self, inputs, conv_output, reward_scaling):
    self.heads = [
        ActorCriticHead('head%d' % i, inputs, conv_output, reward_scaling,
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
    with tf.name_scope(name):
      return self.choose_from_actions(self._log_policy, action)

  def choose_from_actions(self, actions, action):
    return tf.reduce_sum(
        actions * tf.one_hot(action, self.config.num_actions), axis=2)

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
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope.name)

  def activation_summary(self, tensor):
    if self.write_summaries:
      tensor_name = tensor.op.name
      tf.summary.histogram(tensor_name + '/activations', tensor)
      tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(tensor))


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
          tf.multinomial(self.log_policy, num_samples=1),
          axis=1,
          name='greedy_action')
