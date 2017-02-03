import math
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
        conv1 = self.conv_layer(nhwc_input_frames, 8, 8, 4, 32)

      with tf.variable_scope('conv2'):
        conv2 = self.conv_layer(conv1, 4, 4, 2, 64)

      with tf.variable_scope('conv3'):
        conv3 = self.conv_layer(conv2, 3, 3, 1, 64)
        conv3 = tf.reshape(conv3, [-1, 7 * 7 * 64])

      with tf.variable_scope('output'):
        self.action_values = self.action_values_layer(conv3, config)

        self.taken_action_value = tf.reduce_sum(
            self.action_values *
            tf.one_hot(self.action_input, config.num_actions),
            axis=1)

        max_values, max_actions = tf.nn.top_k(self.action_values, k=1)
        self.max_value = tf.squeeze(max_values, axis=1, name='max_value')
        self.max_action = tf.squeeze(max_actions, axis=1, name='max_action')

  def action_values_layer(self, inputs, config):
    if config.dueling:
      # Rescale gradients entering the last convolution layer
      with inputs.graph.gradient_override_map({'Identity': 'grad_over_sqrt2'}):
        inputs = tf.identity(inputs)

      hidden_value = self.fully_connected(
          inputs, 512, activation_fn=tf.nn.relu, name='hidden_value')
      value = self.fully_connected(
          hidden_value, 1, activation_fn=tf.nn.relu, name='value')

      hidden_actions = self.fully_connected(
          inputs, 512, activation_fn=tf.nn.relu, name='hidden_actions')
      actions = self.fully_connected(
          hidden_actions,
          config.num_actions,
          activation_fn=tf.nn.relu,
          name='actions')

      return tf.identity(
          value + actions - tf.reduce_mean(
              actions, axis=1, keep_dims=True),
          name='action_values')

    else:
      hidden = self.fully_connected(
          inputs, 512, activation_fn=tf.nn.relu, name='hidden')

      return self.fully_connected(
          hidden, config.num_actions, activation_fn=None, name='action_values')

  def conv_layer(self, inputs, height, width, stride, filters):
    kernel_shape = [height, width, inputs.get_shape().as_list()[-1], filters]
    kernel = util.variable_with_weight_decay('weights', kernel_shape)
    biases = util.variable_on_cpu('bias', [filters],
                                  tf.constant_initializer(0.1))

    conv = tf.nn.conv2d(
        inputs, kernel, strides=[1, stride, stride, 1], padding='VALID')
    bias = tf.nn.bias_add(conv, biases)
    relu = tf.nn.relu(bias)

    util.activation_summary(relu)
    return relu

  def fully_connected(self, inputs, size, activation_fn, name=None):
    with tf.variable_scope(name):
      weights_shape = [inputs.get_shape().as_list()[-1], size]
      weights = util.variable_with_weight_decay('weights', weights_shape)
      biases = util.variable_on_cpu('bias', [size],
                                    tf.constant_initializer(0.1))

      logits = tf.nn.xw_plus_b(inputs, weights, biases)
      if activation_fn:
        output = activation_fn(logits, name=name)
      else:
        output = tf.identity(logits, name=name)

      util.activation_summary(output)
      return output

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


class PolicyNetwork(QNetwork):
  def __init__(self, config, reuse=None, input_frames=None):
    self.scope = 'policy'
    super(PolicyNetwork, self).__init__(self.scope, reuse, config,
                                        input_frames)


class TargetNetwork(QNetwork):
  def __init__(self, config, reuse=None, input_frames=None, action_input=None):
    self.scope = 'target'
    super(TargetNetwork, self).__init__(self.scope, reuse, config,
                                        input_frames, action_input)
    if config.double_q:
      self.setup_double_q_max_value(config)

    self.reward_input = tf.placeholder(tf.float32, [None], name='reward')
    self.alive_input = tf.placeholder(tf.float32, [None], name='alive')

    self.target_action_value = self.reward_input + (
        self.alive_input * config.discount_rate * self.max_value)

  def setup_double_q_max_value(self, config):
    self.policy_network = PolicyNetwork(
        config, reuse=True, input_frames=self.input_frames)

    with tf.variable_scope('double_q'):
      self.max_action = tf.identity(
          self.policy_network.max_action, name='max_action')

      max_value = (tf.one_hot(self.max_action, config.num_actions) *
                   self.action_values)
      self.max_value = tf.reduce_sum(max_value, axis=1, name='max_value')

  def square_error(self, policy_network):
    # Use tf.stop_gradient to only update the action-value network,
    # not the target action-value network
    square_error = tf.square(
        tf.stop_gradient(self.target_action_value) -
        policy_network.taken_action_value)

    return square_error


@tf.RegisterGradient('grad_over_sqrt2')
def grad_over_sqrt2(op, grad):
  return grad / math.sqrt(2.0)
