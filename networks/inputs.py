import numpy as np
import tensorflow as tf
import util


class Inputs(object):
  def __init__(self, config):
    self.offset_inputs = {}

    with tf.name_scope('inputs') as self.scope:
      self.global_step = tf.contrib.framework.get_or_create_global_step()

      self.replay_count = auto_placeholder(tf.int32, (), 'replay_count',
                                           lambda memory, _: memory.count)

      self.frames = auto_placeholder(
          dtype=tf.uint8,
          shape=[config.input_frames] + list(config.input_shape),
          name='frames',
          feed_data=lambda memory, indices: memory.frames[indices],
          # Centre around 0, scale between [-1, 1]
          preprocess_offset=lambda frames: (tf.to_float(frames) / 127.5) - 1)

      self.actions = auto_placeholder(
          tf.int32, [1], 'actions',
          lambda memory, indices: memory.actions[indices])

      self.rewards = auto_placeholder(
          tf.float32, [1], 'rewards',
          lambda memory, indices: memory.rewards[indices])

      self.alives = auto_placeholder(
          tf.float32, [1], 'alives',
          lambda memory, indices: memory.alives[indices])

      self.discounted_rewards = auto_placeholder(
          tf.float32, [1], 'discounted_rewards',
          lambda memory, indices: memory.discounted_rewards[indices])

      self.bootstrap_mask = auto_placeholder(
          tf.float32, [1], 'bootstrap_mask',
          lambda memory, indices: memory.bootstrap_mask[indices])

      self.priority_probabilities = auto_placeholder(
          tf.float32, [1], 'priority_probabilities',
          lambda memory, indices: memory.priorities.probabilities(indices))

  def offset_input(self, t):
    if t not in self.offset_inputs:
      with tf.name_scope(self.scope):
        with tf.name_scope(util.format_offset('input', t)):
          offset_input = OffsetInput(self, t)
          self.offset_inputs[t] = offset_input
    return self.offset_inputs[t]


def auto_placeholder(dtype, shape, name, feed_data, preprocess_offset=None):
  placeholder_shape = [None, None] + list(shape)[1:] if shape else shape
  placeholder = tf.placeholder(dtype, placeholder_shape, name)
  placeholder.required_feeds = RequiredFeeds(placeholder)
  placeholder.feed_data = feed_data

  tensor = preprocess_offset(placeholder) if preprocess_offset else placeholder

  def offset_data(t, name):
    input_len = shape[0]
    if not hasattr(placeholder, 'zero_offset'):
      placeholder.zero_offset = tf.placeholder_with_default(
          input_len - 1,  # If no zero_offset is given assume that t = 0
          (),
          name + '/zero_offset')

    end = t + 1
    start = end - input_len
    zero_offset = placeholder.zero_offset
    offset_tensor = tensor[:, start + zero_offset:end + zero_offset]

    input_range = np.arange(start, end)
    offset_tensor.required_feeds = RequiredFeeds(placeholder, input_range)

    return tf.reshape(offset_tensor, [-1] + shape, name)

  placeholder.offset_data = offset_data
  return placeholder


class OffsetInput(object):
  def __init__(self, inputs, t):
    # Only use for passing in full observation
    if t == 0:
      self.observations = inputs.frames

    self.frames = inputs.frames.offset_data(t, 'frames')
    self.action = inputs.actions.offset_data(t, 'action')
    self.reward = inputs.rewards.offset_data(t, 'reward')
    self.alive = inputs.alives.offset_data(t, 'alive')
    self.discounted_reward = inputs.discounted_rewards.offset_data(
        t, 'discounted_reward')


class RequiredFeeds(object):
  def __init__(self, placeholder=None, time_offsets=0, feeds=None):
    self.feeds = feeds or {}
    if placeholder is None:
      return

    if isinstance(time_offsets, int):
      time_offsets = np.arange(time_offsets, time_offsets + 1)
    self.feeds[placeholder] = time_offsets

  def merge(self, other):
    if not self.feeds:
      return other
    elif not other.feeds:
      return self

    feeds = {}
    keys = set(self.feeds.keys()) | set(other.feeds.keys())
    for key in keys:
      if key in self.feeds and key in other.feeds:
        full_range = list(self.feeds[key]) + list(other.feeds[key])
        feeds[key] = np.arange(min(full_range), max(full_range) + 1)
      elif key in self.feeds:
        feeds[key] = self.feeds[key]
      else:
        feeds[key] = other.feeds[key]

    return RequiredFeeds(feeds=feeds)

  def input_range(self):
    offsets = set()
    for value in self.feeds.values():
      offsets |= set(value)
    return np.arange(min(offsets), max(offsets))

  def feed_dict(self, indices, replay_memory):
    indices = indices.reshape(-1, 1)
    feed_dict = {}
    for feed, input_range in self.feeds.items():
      offset_indices = replay_memory.offset_index(indices, input_range)
      feed_dict[feed] = feed.feed_data(replay_memory, offset_indices)
      if hasattr(feed, 'zero_offset'):
        feed_dict[feed.zero_offset] = -min(input_range)

    return feed_dict

  @classmethod
  def required_feeds(cls, tensor):
    if hasattr(tensor, 'required_feeds'):
      # Return cached result
      return tensor.required_feeds
    # Get feeds required by all inputs
    if isinstance(tensor, list):
      input_tensors = tensor
    else:
      op = tensor if isinstance(tensor, tf.Operation) else tensor.op
      input_tensors = list(op.inputs) + list(op.control_inputs)

    from networks import inputs
    feeds = inputs.RequiredFeeds()
    for input_tensor in input_tensors:
      feeds = feeds.merge(cls.required_feeds(input_tensor))

    # Cache results
    if not isinstance(tensor, list):
      tensor.required_feeds = feeds

    return feeds
