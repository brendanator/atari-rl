import tensorflow as tf
from . import util

tf.app.flags.DEFINE_integer('input_height', 84, 'Rescale input to this height')
tf.app.flags.DEFINE_integer('input_width', 84, 'Rescale input to this width')
tf.app.flags.DEFINE_integer('input_frames', 4, 'Number of frames to input')
tf.app.flags.DEFINE_float('discount_factor', 0.99,
                          'Discount factor for future rewards')


def deep_q_network(config, num_actions):
  with tf.variable_scope('input_frames'):
    input_frames = tf.placeholder(tf.float32, [
        None, config.input_frames, config.input_height, config.input_width
    ], 'input_frames')
    nhwc_input_frames = tf.transpose(input_frames, [0, 2, 3, 1])

  with tf.variable_scope('conv1'):
    conv1 = conv_layer(nhwc_input_frames, 8, 8, 4, 32)

  with tf.variable_scope('conv2'):
    conv2 = conv_layer(conv1, 4, 4, 2, 64)

  with tf.variable_scope('conv3'):
    conv3 = conv_layer(conv2, 3, 3, 1, 64)

  with tf.variable_scope('fully_connected'):
    flattened = tf.reshape(conv3, [-1, 7 * 7 * 64])
    hidden = fully_connected(flattened, 512, activation_fn=tf.nn.relu)

  with tf.variable_scope('output'):
    action_values = fully_connected(
        hidden, num_actions, activation_fn=None, name='action_values')
    max_values, max_actions = tf.nn.top_k(action_values, k=1)
    max_values = tf.squeeze(max_values, axis=1, name='max_values')
    max_action = tf.squeeze(max_actions, name='max_action')

  return input_frames, action_values, max_values, max_action


def conv_layer(inputs, height, width, stride, filters):
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


def fully_connected(inputs, size, activation_fn, name=None):
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


def loss(target_max_values, action_values, config):
  with tf.variable_scope('loss'):
    reward_input = tf.placeholder(tf.float32, [None], name='reward')
    action_input = tf.placeholder(tf.int32, [None], name='action')
    done_input = tf.placeholder(tf.bool, [None], name='done')

    target_action_value = tf.select(
        done_input, reward_input,
        reward_input + config.discount_factor * target_max_values)
    predicted_action_value = tf.reduce_sum(
        action_values * tf.one_hot(action_input, config.num_actions), axis=1)

    # Use tf.stop_gradient to only update the action-value network,
    # not the target action-value network
    square_error = tf.square(
        tf.stop_gradient(target_action_value) - predicted_action_value)
    loss = tf.reduce_mean(square_error)

  return reward_input, action_input, done_input, loss


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


def copy_q_network(from_scope, to_scope, name):
  """Copy the tensor variables with the same names between the two scopes"""

  named_variables = {var.name: var for var in tf.trainable_variables()}

  operations = []
  for var_name, variable in named_variables.items():
    prefix, suffix = var_name.split('/', 1)
    if prefix == from_scope.name:
      to_variable = named_variables[to_scope.name + '/' + suffix]
      operations.append(to_variable.assign(variable).op)

  with tf.control_dependencies(operations):
    name = name or 'copy_q_network'
    return tf.no_op(name=name)
