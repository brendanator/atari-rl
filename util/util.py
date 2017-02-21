from datetime import datetime
import numpy as np
import scipy.misc
import tensorflow as tf
import time

LUMINANCE_RATIOS = [0.2126, 0.7152, 0.0722]
GRADIENT_SCALING = 'gradient_scaled_by_'


def process_image(frame1, frame2, shape):
  # Max last 2 frames to remove flicker
  image = np.stack([frame1, frame2], axis=3).max(axis=3)

  # Rescale image
  image = scipy.misc.imresize(image, shape)

  # Convert to greyscale
  image = (LUMINANCE_RATIOS * image).sum(axis=2)

  return image


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


def log_episode(episode, start_time, score, steps):
  duration = time.time() - start_time
  steps_per_sec = steps / duration
  format_string = 'Episode %d, score %.0f (%d steps, %.2f secs, %.2f steps/sec)'
  log(format_string % (episode, score, steps, duration, steps_per_sec))


def memoize(f):
  """ Memoization decorator for a function taking one or more arguments.
    Taken from here: https://code.activestate.com/recipes/578231-probably-the-fastest-memoization-decorator-in-the-/#c4
    """

  class memodict(dict):
    def __getitem__(self, *key):
      return dict.__getitem__(self, key)

    def __missing__(self, key):
      self[key] = ret = f(*key)
      return ret

  return memodict().__getitem__
