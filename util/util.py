from datetime import datetime
import numpy as np
import os
import tensorflow as tf
import time

LUMINANCE_RATIOS = [0.2126, 0.7152, 0.0722]
GRADIENT_SCALING = 'gradient_scaled_by_'


def run_directory(config):
  def find_previous_run(dir):
    if os.path.isdir(dir):
      runs = [child[4:] for child in os.listdir(dir) if child[:4] == 'run_']
      if runs:
        return max([int(run) for run in runs])

    return 0

  if config.run_dir == 'latest':
    parent_dir = 'runs/%s/' % config.game
    previous_run = find_previous_run(parent_dir)
    run_dir = parent_dir + ('run_%d' % previous_run)
  elif config.run_dir:
    run_dir = config.run_dir
  else:
    parent_dir = 'runs/%s/' % config.game
    previous_run = find_previous_run(parent_dir)
    run_dir = parent_dir + ('run_%d' % (previous_run + 1))

  if run_dir[-1] != '/':
    run_dir += '/'

  if not os.path.isdir(run_dir):
    os.makedirs(run_dir)

  log('Checkpoints and summaries will be written to %s' % run_dir)

  return run_dir


def format_offset(prefix, t):
  if t > 0:
    return prefix + '_t_plus_' + str(t)
  elif t == 0:
    return prefix + '_t'
  else:
    return prefix + '_t_minus_' + str(-t)


def add_loss_summaries(total_loss):
  """Add summaries for losses in model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar('loss_raw', l)
    tf.summary.scalar('loss', loss_averages.average(l))

  return loss_averages_op


def activation_summary(x):
  """Helper to create summaries for activations.
  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.
  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  # tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tensor_name = x.op.name
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


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
