import tensorflow as tf
from . import util


class QNetwork:
  def __init__(self, scope_name, reuse, config):
    with tf.variable_scope(scope_name, reuse=reuse):
      with tf.variable_scope('input_frames'):
        self.input_frames = tf.placeholder(tf.float32, [
            None, config.input_frames, config.input_height, config.input_width
        ], 'input_frames')
        nhwc_input_frames = tf.transpose(self.input_frames, [0, 2, 3, 1])

      with tf.variable_scope('conv1'):
        conv1 = self.conv_layer(nhwc_input_frames, 8, 8, 4, 32)

      with tf.variable_scope('conv2'):
        conv2 = self.conv_layer(conv1, 4, 4, 2, 64)

      with tf.variable_scope('conv3'):
        conv3 = self.conv_layer(conv2, 3, 3, 1, 64)

      with tf.variable_scope('fully_connected'):
        flattened = tf.reshape(conv3, [-1, 7 * 7 * 64])
        hidden = self.fully_connected(flattened, 512, activation_fn=tf.nn.relu)

      with tf.variable_scope('output'):
        self.action_values = self.fully_connected(
            hidden,
            config.num_actions,
            activation_fn=None,
            name='action_values')
        max_values, max_actions = tf.nn.top_k(self.action_values, k=1)
        self.max_values = tf.squeeze(max_values, axis=1, name='max_values')
        self.max_action = tf.squeeze(max_actions, name='max_action')

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
    weights_shape = [inputs.get_shape().as_list()[-1], size]
    weights = util.variable_with_weight_decay('weights', weights_shape)
    biases = util.variable_on_cpu('bias', [size], tf.constant_initializer(0.1))

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
  def __init__(self, config):
    self.scope = 'policy'
    super(PolicyNetwork, self).__init__(self.scope, False, config)


class TargetNetwork(QNetwork):
  def __init__(self, config):
    self.scope = 'target'
    super(TargetNetwork, self).__init__(self.scope, False, config)


def loss(policy_network, target_network, config):
  with tf.variable_scope('loss'):
    action_input = tf.placeholder(tf.int32, [None], name='action')
    reward_input = tf.placeholder(tf.float32, [None], name='reward')
    done_input = tf.placeholder(tf.float32, [None], name='done')

    target_action_value = (reward_input + (1.0 - done_input) *
                           config.discount_factor * target_network.max_values)

    predicted_action_value = tf.reduce_sum(
        policy_network.action_values *
        tf.one_hot(action_input, config.num_actions),
        axis=1)

    # Use tf.stop_gradient to only update the action-value network,
    # not the target action-value network
    square_error = tf.square(
        tf.stop_gradient(target_action_value) - predicted_action_value)
    loss = tf.reduce_mean(square_error)

  return action_input, reward_input, done_input, loss


def train(loss, global_step):
  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = util.add_loss_summaries(loss)

  # Optimizer
  opt = tf.train.AdamOptimizer()

  # Minimize loss
  with tf.control_dependencies([loss_averages_op]):
    grads = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram('trainable', var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram('gradient', grad)

  return train_op
