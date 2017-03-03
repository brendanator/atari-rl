from datetime import datetime
import numpy as np
import re
import scipy.misc
import tensorflow as tf
import time

LUMINANCE_RATIOS = [0.2126, 0.7152, 0.0722]
GRADIENT_SCALING = 'gradient_scaled_by_'
TOWER_NAME = 'TOWER'


def process_image(frame1, frame2, shape):
  # Max last 2 frames to remove flicker
  image = np.stack([frame1, frame2], axis=3).max(axis=3)

  # Rescale image
  image = scipy.misc.imresize(image, shape)

  # Convert to greyscale
  image = (LUMINANCE_RATIOS * image).sum(axis=2)

  return image


def add_loss_summaries(total_loss, scope):
  """Add summaries for losses in model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  # TODO loss_averages does not exist because of the variable_scope being reused
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses', scope)
  print(losses, total_loss)
  # TODO
  # loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    loss_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', l.op.name)

    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(loss_name + '_raw', l)
    # tf.summary.scalar(loss_name, loss_averages.average(l))

  # return loss_averages_op


def variable_on_cpu(name, shape, initializer):
  with tf.device('/gpu:0'):
    return tf.get_variable(name, shape, initializer=initializer)


def conv2d(inputs, filters, kernel_size, strides, name):
  with tf.variable_scope(name):
    input_dim = inputs.get_shape()[-1]
    kernel_shape = kernel_size + [input_dim, filters]
    kernel = variable_on_cpu('weights', shape=kernel_shape, initializer=None)
    strides = [1] + strides + [1]
    biases = variable_on_cpu('biases', [filters], tf.constant_initializer(0.0))

    conv = tf.nn.conv2d(inputs, kernel, strides, padding='VALID')
    bias = tf.nn.bias_add(conv, biases)
    return tf.nn.relu(bias, name=name)


def dense(inputs, units, activation=None, name=None):
  with tf.variable_scope(name):
    input_dim = inputs.get_shape()[-1]
    weights_shape = [input_dim, units]
    weights = variable_on_cpu('weights', shape=weights_shape, initializer=None)
    biases = variable_on_cpu('biases', [units], tf.constant_initializer(0.0))

    output = tf.matmul(inputs, weights)
    if activation:
      return activation(output + biases, name=name)
    else:
      return tf.nn.bias_add(output, biases, name=name)


def scale_gradient(inputs, scale):
  with inputs.graph.gradient_override_map({'Identity': 'scaled_gradient'}):
    return tf.identity(inputs, name=GRADIENT_SCALING + str(scale))


@tf.RegisterGradient('scaled_gradient')
def scaled_gradient(op, grad):
  scale_index = op.name.rfind(GRADIENT_SCALING) + len(GRADIENT_SCALING)
  scale = float(op.name[scale_index:])
  return grad * scale


def log(message):
  import threading
  thread_id = threading.current_thread().name
  now = datetime.strftime(datetime.now(), '%F %X')
  print('%s %s: %s' % (now, thread_id, message))


def memoize(f):
  """ Memoization decorator for a function taking one or more arguments.
    Taken from here: https://goo.gl/gxOVPQ
    """

  class memodict(dict):
    def __getitem__(self, *key):
      return dict.__getitem__(self, key)

    def __missing__(self, key):
      self[key] = ret = f(*key)
      return ret

  return memodict().__getitem__
