import numpy as np
import tensorflow as tf
import util


class Inputs(object):
  def __init__(self, config):
    self.config = config
    self.offset_inputs = {}

    with tf.name_scope('inputs') as self.scope:
      self.global_step = tf.contrib.framework.get_or_create_global_step()

      self.replay_count = tf.placeholder(tf.int32, (), 'replay_count')
      self.replay_count.required_feeds = RequiredFeeds(self.replay_count)
      self.replay_count.feed_data = lambda memory, _: memory.count

      shape = [None, None] + list(config.input_shape)
      self.frames = tf.placeholder(tf.uint8, shape, 'frames')
      self.frames.zero_offset = tf.placeholder_with_default(
          1 - config.input_frames, ())
      self.frames.feed_data = lambda memory, indices: memory.frames[indices]

      self.actions = tf.placeholder(tf.int32, [None, None], name='actions')
      self.actions.zero_offset = tf.placeholder(tf.int32, ())
      self.actions.feed_data = lambda memory, indices: memory.actions[indices]

      self.rewards = tf.placeholder(tf.float32, [None, None], name='rewards')
      self.rewards.zero_offset = tf.placeholder(tf.int32, ())
      self.rewards.feed_data = lambda memory, indices: memory.rewards[indices]

      self.alives = tf.placeholder(tf.float32, [None, None], name='alives')
      self.alives.zero_offset = tf.placeholder(tf.int32, ())
      self.alives.feed_data = lambda memory, indices: memory.alives[indices]

      self.total_rewards = tf.placeholder(
          tf.float32, [None, None], name='total_rewards')
      self.total_rewards.zero_offset = tf.placeholder(tf.int32, ())
      self.total_rewards.feed_data = (
          lambda memory, indices: memory.total_rewards[indices])

      self.bootstrap_mask = tf.placeholder(
          tf.float32, [None, config.num_bootstrap_heads],
          name='bootstrap_mask')
      self.bootstrap_mask.required_feeds = RequiredFeeds(self.bootstrap_mask)
      self.bootstrap_mask.feed_data = (
          lambda memory, indices: memory.bootstrap_mask[indices])

      self.priority_probs = tf.placeholder(tf.float32, [None, None],
                                           'priority_probs')
      self.priority_probs.required_feeds = RequiredFeeds(self.priority_probs)
      self.priority_probs.feed_data = (
          lambda memory, indices: memory.priorities.probabilities(indices))

  def offset_input(self, t):
    if t not in self.offset_inputs:
      with tf.name_scope(self.scope):
        with tf.name_scope(util.format_offset('input', t)):
          offset_input = OffsetInput(self, t, self.config)
          self.offset_inputs[t] = offset_input
    return self.offset_inputs[t]


class OffsetInput(object):
  def __init__(self, inputs, t, config):
    first_frame = 1 - config.input_frames
    t_offset = inputs.frames.zero_offset + t
    start_t_offset = t_offset + first_frame
    frames = inputs.frames[:, start_t_offset:t_offset + 1]
    required_range = np.arange(first_frame + t, t + 1)
    frames.required_feeds = RequiredFeeds(inputs.frames, required_range)
    shape = [-1, config.input_frames] + list(config.input_shape)
    self.observations = tf.reshape(frames, shape, name='observations')
    # Centre around 0, scale between [-1, 1]
    self.frames = (tf.to_float(self.observations) / 127.5) - 1

    # Inputs are expanded to shape [None, 1] to allow broadcasting
    #   and avoid operations creating shapes like [None, None, ...]
    action = inputs.actions[:, inputs.actions.zero_offset + t]
    action.required_feeds = RequiredFeeds(inputs.actions, t)
    self.action = tf.expand_dims(action, axis=1, name='action')

    reward = inputs.rewards[:, inputs.rewards.zero_offset + t]
    reward.required_feeds = RequiredFeeds(inputs.rewards, t)
    self.reward = tf.expand_dims(reward, axis=1, name='reward')

    alive = inputs.alives[:, inputs.alives.zero_offset + t]
    alive.required_feeds = RequiredFeeds(inputs.alives, t)
    self.alive = tf.expand_dims(alive, axis=1, name='alive')

    total_reward = inputs.total_rewards[:, inputs.total_rewards.zero_offset +
                                        t]
    total_reward.required_feeds = RequiredFeeds(inputs.total_rewards, t)
    self.total_reward = tf.expand_dims(
        total_reward, axis=1, name='total_reward')


class RequiredFeeds(object):
  def __init__(self, placeholder=None, time_offsets=0, feeds=None):
    if feeds:
      self.feeds = feeds
    else:
      self.feeds = {}

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
    else:
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
