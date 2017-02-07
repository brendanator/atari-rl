import math
import numpy as np
import tensorflow as tf

from . import util


class QNetwork:
  def __init__(self,
               scope_name,
               reuse,
               config,
               input_frames=None,
               action_input=None):

    with tf.variable_scope(scope_name, reuse=reuse):
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

      with tf.variable_scope('conv1'):
        conv1 = util.conv_layer(nhwc_input_frames, 8, 8, 4, 32)

      with tf.variable_scope('conv2'):
        conv2 = util.conv_layer(conv1, 4, 4, 2, 64)

      with tf.variable_scope('conv3'):
        conv3 = util.conv_layer(conv2, 3, 3, 1, 64)
        conv_output = tf.reshape(conv3, [-1, 7 * 7 * 64])

      self.num_heads = config.num_bootstrap_heads if config.bootstrapped else 1

      if self.num_heads > 1:
        # Rescale gradients entering the last convolution layer
        conv_output = util.scale_gradient(
            conv_output, scale=1 / self.num_heads)

      self.heads = [
          OutputHead(conv_output, self.action_input, 'head-%d' % i, config)
          for i in range(self.num_heads)
      ]

      self.heads_action_values = tf.pack(
          [head.action_values for head in self.heads], axis=1)

      self.heads_taken_action_value = tf.pack(
          [head.taken_action_value for head in self.heads], axis=1)

      self.heads_max_action = tf.pack(
          [head.max_action for head in self.heads], axis=1)

      self.using_ensemble = config.bootstrap_use_ensemble
      if self.using_ensemble:
        ensemble_votes = tf.reduce_sum(
            tf.one_hot(self.heads_max_action, config.num_actions), axis=1)
        # Add some noise to break ties
        noise = tf.random_uniform([config.num_actions])
        _, ensemble_max_action = tf.nn.top_k(ensemble_votes + noise, k=1)
        self.ensemble_max_action = tf.squeeze(
            ensemble_max_action, axis=1, name='ensemble_max_action')

  def sample_head(self):
    self.active_head = self.heads[np.random.randint(self.num_heads)]

  @property
  def max_action(self):
    if self.using_ensemble:
      return self.ensemble_max_action
    else:
      return self.active_head.max_action

  def copy_to_network(self, to_network):
    """Copy the tensor variables with the same names between the two scopes"""

    named_variables = {var.name: var for var in tf.trainable_variables()}

    operations = []
    for var_name, variable in named_variables.items():
      prefix, suffix = var_name.split('/', 1)
      if prefix == self.scope:
        to_variable = named_variables[to_network.scope + '/' + suffix]
        operations.append(to_variable.assign(variable).op)

    with tf.control_dependencies(operations):
      return tf.no_op(name='copy_q_network')


class OutputHead:
  def __init__(self, inputs, action_input, name, config):
    with tf.variable_scope(name):
      self.action_values = self.action_values_layer(inputs, config)

      # Policy only
      self.taken_action_value = tf.reduce_sum(
          self.action_values * tf.one_hot(action_input, config.num_actions),
          axis=1)

      max_values, max_actions = tf.nn.top_k(self.action_values, k=1)
      # Target only
      self.max_value = tf.squeeze(max_values, axis=1, name='max_value')
      # Policy only
      self.max_action = tf.squeeze(max_actions, axis=1, name='max_action')

  def action_values_layer(self, inputs, config):
    if config.dueling:
      # Rescale gradients entering the last convolution layer
      inputs = util.scale_gradient(inputs, scale=1 / math.sqrt(2))

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

      return tf.identity(
          value + actions - tf.reduce_mean(
              actions, axis=1, keep_dims=True),
          name='action_values')

    else:
      hidden = util.fully_connected(
          inputs, 512, activation_fn=tf.nn.relu, name='hidden')

      return util.fully_connected(
          hidden, config.num_actions, activation_fn=None, name='action_values')


class PolicyNetwork(QNetwork):
  def __init__(self, config, reuse=None, input_frames=None):
    self.scope = 'policy'
    super(PolicyNetwork, self).__init__(self.scope, reuse, config,
                                        input_frames)


class TargetNetwork(QNetwork):
  def __init__(self,
               policy_network,
               config,
               reuse=None,
               input_frames=None,
               action_input=None):
    self.config = config
    self.policy_network = policy_network
    self.scope = 'target'
    super(TargetNetwork, self).__init__(self.scope, reuse, config,
                                        input_frames, action_input)

    if config.double_q:
      with tf.variable_scope('double_q'):
        self.heads_max_action = tf.identity(
            policy_network.heads_max_action, name='heads_max_action')

        self.heads_max_value = tf.reduce_sum(
            tf.one_hot(self.heads_max_action, config.num_actions) *
            self.heads_action_values,
            axis=2,
            name='heads_max_value')
    else:
      self.heads_max_value = tf.pack(
          [head.max_value for head in self.heads],
          axis=1,
          name='heads_max_value')

    self.reward_input = tf.placeholder(tf.float32, [None], name='reward')
    self.alive_input = tf.placeholder(tf.float32, [None], name='alive')
    reward_input = tf.expand_dims(self.reward_input, axis=1)
    alive_input = tf.expand_dims(self.alive_input, axis=1)

    self.heads_target_action_value = reward_input + (
        alive_input * config.discount_rate * self.heads_max_value)

  def square_errors(self, policy_network):
    # Use tf.stop_gradient to only update the policy action-value network,
    # not the target action-value network
    square_errors = tf.square(
        tf.stop_gradient(self.heads_target_action_value) -
        policy_network.heads_taken_action_value,
        name='square_errors')

    if self.config.bootstrapped and self.config.bootstrap_mask_probability < 1.0:
      self.bootstrap_mask = tf.placeholder(
          tf.float32, [None, self.num_heads], name='bootstrap_mask')
      square_errors *= self.bootstrap_mask

    return square_errors
