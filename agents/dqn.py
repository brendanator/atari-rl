import math
import numpy as np
import tensorflow as tf

from . import util


class QNetwork(object):
  def __init__(self,
               scope,
               reward_scaling,
               config,
               reuse=None,
               input_frames=None,
               action_input=None):

    self.scope = scope
    self.num_heads = config.num_bootstrap_heads if config.bootstrapped else 1
    self.using_ensemble = config.bootstrap_use_ensemble

    with tf.variable_scope(scope, reuse=reuse):
      input_frames, action_input = self.build_inputs(input_frames,
                                                     action_input, config)

      conv_output = self.build_conv_layers(input_frames, config)

      self.build_heads(conv_output, reward_scaling, config)

      self.sample_head()

  def build_inputs(self, input_frames, action_input, config):
    with tf.variable_scope('input'):
      # Input frames
      if input_frames is None:
        self.input_frames = tf.placeholder(
            tf.float32, [None, config.input_frames] + config.input_shape,
            'input_frames')
      else:
        # Reuse another QNetwork's input_frames
        self.input_frames = input_frames
      nhwc_input_frames = tf.transpose(self.input_frames, [0, 2, 3, 1])

      # Taken actions
      if action_input is None:
        self.action_input = tf.placeholder(tf.int32, [None], name='action')
      else:
        self.action_input = action_input

    return nhwc_input_frames, action_input

  def build_conv_layers(self, input_frames, config):
    with tf.variable_scope('conv1'):
      conv1 = util.conv_layer(input_frames, 8, 8, 4, 32)

    with tf.variable_scope('conv2'):
      conv2 = util.conv_layer(conv1, 4, 4, 2, 64)

    with tf.variable_scope('conv3'):
      conv3 = util.conv_layer(conv2, 3, 3, 1, 64)
      conv_output = tf.reshape(conv3, [-1, 7 * 7 * 64])

    # Rescale gradients entering the last convolution layer
    scale = (1.0 / math.sqrt(2) if config.dueling else 1.0) / self.num_heads
    if scale < 1:
      conv_output = util.scale_gradient(conv_output, scale)

    return conv_output

  def build_heads(self, conv_output, reward_scaling, config):
    self.heads = [
        ActionValueHead('head-%d' % i, conv_output, reward_scaling, config)
        for i in range(self.num_heads)
    ]

    self.action_values = tf.stack(
        [head.action_value for head in self.heads], axis=1)

    action_input = tf.expand_dims(self.action_input, axis=1)
    self.taken_action_values = tf.reduce_sum(
        self.action_values * tf.one_hot(action_input, config.num_actions),
        axis=2,
        name='taken_action_values')

    values, max_actions = tf.nn.top_k(self.action_values, k=1)
    self.values = tf.squeeze(values, axis=2, name='values')
    self.max_actions = tf.squeeze(max_actions, axis=2, name='max_actions')

    if self.using_ensemble:
      ensemble_votes = tf.reduce_sum(
          tf.one_hot(self.max_actions, config.num_actions), axis=1)
      # Add some noise to break ties
      noise = tf.random_uniform([config.num_actions])
      _, ensemble_max_action = tf.nn.top_k(ensemble_votes + noise, k=1)
      self.ensemble_max_action = tf.squeeze(
          ensemble_max_action, axis=1, name='ensemble_max_action')

  def sample_head(self):
    self.active_head = self.heads[np.random.randint(self.num_heads)]

  @property
  def max_action(self):
    if self.num_heads == 1 or not self.using_ensemble:
      return self.active_head.max_action
    else:
      return self.ensemble_max_action

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
      action_value = self.action_value_layer(inputs, config)

      if reward_scaling:
        action_value = reward_scaling.unnormalize_output(action_value)

      self.action_value = tf.identity(action_value, name='action_value')

      value, max_action = tf.nn.top_k(self.action_value, k=1)
      self.value = tf.squeeze(value, axis=1, name='value')
      self.max_action = tf.squeeze(max_action, axis=1, name='max_action')

  def action_value_layer(self, inputs, config):
    if config.dueling:
      hidden_value = util.fully_connected(
          inputs, 512, activation_fn=tf.nn.relu, name='hidden_value')
      value = util.fully_connected(
          hidden_value, 1, activation_fn=tf.nn.relu, name='value')

      hidden_actions = util.fully_connected(
          inputs, 512, activation_fn=tf.nn.relu, name='hidden_actions')
      actions = util.fully_connected(
          hidden_actions,
          config.num_actions,
          activation_fn=tf.nn.relu,
          name='actions')

      return value + actions - tf.reduce_mean(actions, axis=1, keep_dims=True)

    else:
      hidden = util.fully_connected(
          inputs, 512, activation_fn=tf.nn.relu, name='hidden')

      return util.fully_connected(
          hidden, config.num_actions, activation_fn=None, name='action_value')


class PolicyNetwork(QNetwork):
  def __init__(self, reward_scaling, config):
    super(PolicyNetwork, self).__init__('policy', reward_scaling, config)


class TargetNetwork(QNetwork):
  def __init__(self,
               policy_network,
               reward_scaling,
               config,
               reuse=None,
               input_frames=None,
               action_input=None):
    self.config = config
    super(TargetNetwork, self).__init__(
        scope='target',
        reward_scaling=reward_scaling,
        config=config,
        reuse=reuse,
        input_frames=input_frames,
        action_input=action_input)

    if config.double_q:
      with tf.variable_scope('double_q'):
        # Policy network shouldn't be updated when calculating target values
        self.max_actions = tf.stop_gradient(
            policy_network.max_actions, name='max_actions')

        self.values = tf.reduce_sum(
            tf.one_hot(self.max_actions, config.num_actions) *
            self.action_values,
            axis=2,
            name='values')

    self.reward_input = tf.placeholder(tf.float32, [None], name='reward')
    self.alive_input = tf.placeholder(tf.float32, [None], name='alive')
    reward_input = tf.expand_dims(self.reward_input, axis=1)
    alive_input = tf.expand_dims(self.alive_input, axis=1)

    self.target_action_values = reward_input + (
        alive_input * config.discount_rate * self.values)
