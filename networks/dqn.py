import math
import numpy as np
import tensorflow as tf

import util


class QNetwork(object):
  def __init__(self, scope, inputs, reward_scaling, config, reuse):

    self.scope = scope
    self.num_heads = config.num_bootstrap_heads if config.bootstrapped else 1
    self.using_ensemble = config.bootstrap_use_ensemble
    self.config = config

    with tf.variable_scope(scope, reuse=reuse):
      self.conv_layers = self.build_conv_layers(inputs.input_frames, config)
      self.build_heads(self.conv_layers, inputs.action_input, reward_scaling,
                       config)

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

      # Taken actions
      if action_input is None:
        self.action_input = tf.placeholder(tf.int32, [None], name='action')
      else:
        self.action_input = action_input

    return input_frames, action_input

  def build_conv_layers(self, input_frames, config):
    nhwc_input_frames = tf.transpose(input_frames, [0, 2, 3, 1])
    with tf.variable_scope('conv1'):
      conv1 = util.conv_layer(nhwc_input_frames, 8, 8, 4, 32)

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

  def build_heads(self, conv_output, action_input, reward_scaling, config):
    if config.actor_critic:
      head = ActorCriticHead('actor-critic', conv_output, reward_scaling,
                             config)
      self.heads = [head]
      self.values = head.value
      self.greedy_actions = head.greedy_action
    else:
      self.heads = [
          ActionValueHead('head-%d' % i, conv_output, reward_scaling, config)
          for i in range(self.num_heads)
      ]

      self.action_values = tf.stack(
          [head.action_value for head in self.heads], axis=1)

      action_input = tf.expand_dims(action_input, axis=1)
      self.taken_action_values = self.action_value(
          action_input, name='taken_action_values')

      values, greedy_actions = tf.nn.top_k(self.action_values, k=1)
      self.values = tf.squeeze(values, axis=2, name='values')
      self.greedy_actions = tf.squeeze(
          greedy_actions, axis=2, name='greedy_actions')

      if self.using_ensemble:
        ensemble_votes = tf.reduce_sum(
            tf.one_hot(self.greedy_actions, config.num_actions), axis=1)
        # Add some noise to break ties
        noise = tf.random_uniform([config.num_actions])
        _, ensemble_greedy_action = tf.nn.top_k(ensemble_votes + noise, k=1)
        self.ensemble_greedy_action = tf.squeeze(
            ensemble_greedy_action, axis=1, name='ensemble_greedy_action')

  def action_value(self, action, name=None):
    return tf.reduce_sum(
        self.action_values * tf.one_hot(action, self.config.num_actions),
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
      action_value = self.action_value_layer(inputs, config)

      if reward_scaling:
        action_value = reward_scaling.unnormalize_output(action_value)

      self.action_value = tf.identity(action_value, name='action_value')

      value, greedy_action = tf.nn.top_k(self.action_value, k=1)
      self.value = tf.squeeze(value, axis=1, name='value')
      self.greedy_action = tf.squeeze(greedy_action, axis=1, name='greedy_action')

  def action_value_layer(self, inputs, config):
    if config.dueling:
      hidden_value = util.fully_connected(
          inputs, 256, activation_fn=tf.nn.relu, name='hidden_value')
      value = util.fully_connected(
          hidden_value, 1, activation_fn=None, name='value')

      hidden_actions = util.fully_connected(
          inputs, 256, activation_fn=tf.nn.relu, name='hidden_actions')
      actions = util.fully_connected(
          hidden_actions,
          config.num_actions,
          activation_fn=None,
          name='actions')

      return value + actions - tf.reduce_mean(actions, axis=1, keep_dims=True)

    else:
      hidden = util.fully_connected(
          inputs, 256, activation_fn=tf.nn.relu, name='hidden')

      return util.fully_connected(
          hidden, config.num_actions, activation_fn=None, name='action_value')


class ActorCriticHead(object):
  def __init__(self, name, inputs, reward_scaling, config):
    with tf.variable_scope(name):
      hidden = util.fully_connected(
          inputs, 256, activation_fn=tf.nn.relu, name='hidden')

      self.value = util.fully_connected(
          hidden, 1, activation_fn=None, name='value')
      if reward_scaling:
        self.value = reward_scaling.unnormalize_output(self.value)

      actions = util.fully_connected(
          hidden, config.num_actions, activation_fn=None, name='actions')
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


# class PolicyNetwork(QNetwork):
#   def __init__(self, reward_scaling, config, reuse=None):
#     super(PolicyNetwork, self).__init__(
#         'policy', reward_scaling, config, reuse=reuse)

# class TargetNetwork(QNetwork):
#   def __init__(self,
#                policy_network,
#                reward_scaling,
#                config,
#                reuse=None,
#                input_frames=None,
#                action_input=None):
#     self.config = config
#     super(TargetNetwork, self).__init__(
#         scope='target',
#         reward_scaling=reward_scaling,
#         config=config,
#         reuse=reuse,
#         input_frames=input_frames,
#         action_input=action_input)

#     if config.double_q:
#       with tf.variable_scope('double_q'):
#         # Policy network shouldn't be updated when calculating target values
#         self.max_actions = tf.stop_gradient(
#             policy_network.max_actions, name='max_actions')

#         self.values = tf.reduce_sum(
#             tf.one_hot(self.max_actions, config.num_actions) *
#             self.action_values,
#             axis=2,
#             name='values')
#     elif config.sarsa:
#       with tf.variable_scope('sarsa'):
#         self.values = tf.identity(self.taken_action_values, name='values')
