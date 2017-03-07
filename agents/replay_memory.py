import h5py
import numpy as np
import tensorflow as tf
import threading
import util

from .replay_priorities import ProportionalPriorities, UniformPriorities


class ReplayMemory(object):
  def __init__(self, pre_input_offset, post_input_offset, config):
    # Input offsets that must be valid. Final offset can be safely ignored
    self.input_range = np.arange(pre_input_offset, post_input_offset)

    # Config
    self.capacity = config.replay_capacity
    self.discount_rate = config.discount_rate
    self.recent_only = config.async is not None
    self.run_dir = config.run_dir

    # Track position and count in memory
    self.cursor = -1  # Cursor points to index currently being written
    self.count = 0

    # Store all values in numpy arrays
    self.observations = np.zeros(
        [config.replay_capacity, config.input_frames] +
        list(config.input_shape),
        dtype=np.float32)
    self.actions = np.zeros([config.replay_capacity], dtype=np.int32)
    self.rewards = np.zeros([config.replay_capacity], dtype=np.float32)
    self.total_rewards = np.zeros([config.replay_capacity], dtype=np.float32)

    # Store alive instead of done as it simplifies calculations elsewhere
    self.alives = np.zeros([config.replay_capacity], dtype=np.bool)

    # Might as well populate this upfront for all time
    self.bootstrap_masks = np.random.binomial(
        n=1,
        p=config.bootstrap_mask_probability,
        size=[config.replay_capacity, config.num_bootstrap_heads])

    # Priorities
    if config.replay_priorities == 'uniform':
      self.priorities = UniformPriorities()
    elif config.replay_priorities == 'proportional':
      self.priorities = ProportionalPriorities(config)
    else:
      raise Exception('Unknown replay_priorities: ' + config.replay_priorities)

  def save(self):
    name = self.run_dir + 'replay_' + threading.current_thread().name + '.hdf'
    with h5py.File(name, 'w') as h5f:
      util.log('Saving replay memory')
      for key, value in self.__dict__.items():
        if key == 'priorities':
          priorities_group = h5f.create_group(key)
          for p_key, p_value in self.priorities.__dict__.items():
            priorities_group.create_dataset(p_key, data=p_value)
        else:
          h5f.create_dataset(key, data=value)

  def load(self):
    name = self.run_dir + 'replay_' + threading.current_thread().name + '.hdf'
    try:
      with h5py.File(name, 'r') as h5f:
        util.log('Loading replay memory')
        for key in self.__dict__.keys():
          if key == 'priorities':
            priorities_group = h5f[key]
            for p_key in self.priorities.__dict__.keys():
              self.priorities.__dict__[p_key] = h5f[p_key][:]
              priorities_group.create_dataset(p_key, data=p_value)
          else:
            self.__dict__[key] = h5f[key][:]
      return True
    except:
      return False

  def store_new_episode(self, observation):
    self.cursor = self.offset_index(self.cursor, 1)
    self.count = min(self.count + 1, self.capacity)
    self.observations[self.cursor] = observation
    self.alives[self.cursor] = True

  def store_transition(self, action, reward, done, next_observation):
    self.actions[self.cursor] = action
    self.rewards[self.cursor] = reward
    self.total_rewards[self.cursor] = reward
    self.priorities.update_to_highest_priority(self.cursor)

    self.cursor = self.offset_index(self.cursor, 1)
    self.count = min(self.count + 1, self.capacity)

    self.alives[self.cursor] = not done
    self.observations[self.cursor] = next_observation

    if done:
      # Update total_rewards for episode
      i = self.offset_index(self.cursor, -2)
      while self.alives[i] and i < self.count:
        reward = reward * self.discount_rate + self.rewards[i]
        self.total_rewards[i] = reward
        i = self.offset_index(i, -1)

  def offset_index(self, index, offset):
    return (index + offset) % self.capacity

  def sample_batch(self, batch_size, step):
    if self.recent_only:
      indices = self.recent_indices(batch_size)
    else:
      indices = self.sample_indices(batch_size)

    return SampleBatch(self, indices, step)

  def recent_indices(self, batch_size):
    indices = []
    indices_ranges = np.empty(
        shape=(batch_size, len(self.input_range)), dtype=np.int32)
    indices_ranges.fill(-1)

    # Avoid infinite loop from repeated collisions
    retries, max_retries = 0, len(self.input_range) * batch_size

    index = self.offset_index(self.cursor, -max(self.input_range) - 1)
    while len(indices) < batch_size and retries < max_retries:
      if self.update_indices(index, indices, indices_ranges):
        index = self.offset_index(index, -len(self.input_range))
      else:
        index = self.offset_index(index, -1)
        retries += 1

    return np.array(indices)

  def sample_indices(self, batch_size):
    indices = []
    indices_ranges = np.empty(
        shape=(batch_size, len(self.input_range)), dtype=np.int32)
    indices_ranges.fill(-1)

    # Avoid infinite loop from repeated collisions
    retries, max_retries = 0, batch_size

    while len(indices) < batch_size and retries < max_retries:
      index = self.priorities.sample_index(self.count)
      if not self.update_indices(index, indices, indices_ranges):
        retries += 1

    return np.array(indices)

  def update_indices(self, index, indices, indices_ranges):
    num = len(indices)
    index_range = self.offset_index(index, self.input_range)

    # Reject indices too close to the cursor or to the start/end of episode
    excludes_cursor = self.cursor not in index_range
    within_episode = self.alives[index_range].all()
    if not (excludes_cursor and within_episode):
      return False

    # Reject indices whose range overlaps existing indices ranges
    transposed_index_range = index_range.reshape([1, 1, -1]).transpose()
    if (transposed_index_range == indices_ranges[:num]).any():
      return False

    # Update indices
    indices.append(index)
    indices_ranges[num] = index_range
    return True


class SampleBatch(object):
  def __init__(self, replay_memory, indices, step):
    self.replay_memory = replay_memory
    self.priorities = replay_memory.priorities
    self.indices = indices
    self.step = step
    self.is_valid = len(indices) > 0

  def offset_indices(self, offset):
    return self.replay_memory.offset_index(self.indices, offset)

  def observations(self, offset):
    return self.replay_memory.observations[self.offset_indices(offset)]

  def actions(self, offset):
    return self.replay_memory.actions[self.offset_indices(offset)]

  def rewards(self, offset):
    return self.replay_memory.rewards[self.offset_indices(offset)]

  def alives(self, offset):
    return self.replay_memory.alives[self.offset_indices(offset)]

  def total_rewards(self, offset):
    return self.replay_memory.total_rewards[self.offset_indices(offset)]

  def importance_sampling(self):
    return self.priorities.importance_sampling(
        self.indices, self.replay_memory.count, self.step)

  def update_priorities(self, priorites):
    self.priorities.update_priorities(self.indices, priorites)

  def build_feed_dict(self, fetches):
    if not isinstance(fetches, list):
      fetches = [fetches]

    placeholders = set()
    for fetch in fetches:
      placeholders |= self.placeholders(fetch)

    feed_dict = {}
    for placeholder in placeholders:
      feed_dict[placeholder] = placeholder.feed_data(self)

    return feed_dict

  def placeholders(self, tensor):
    if hasattr(tensor, 'feed_data'):
      # Placeholders are decorated with the 'feed_data' method
      return {tensor}
    elif hasattr(tensor, 'placeholders'):
      # Return cached result
      return tensor.placeholders
    else:
      # Get placeholders used by all inputs
      placeholders = set()
      op = tensor if isinstance(tensor, tf.Operation) else tensor.op
      for input_tensor in op.inputs:
        placeholders |= self.placeholders(input_tensor)
      for control_tensor in op.control_inputs:
        placeholders |= self.placeholders(control_tensor)

      # Cache results
      tensor.placeholders = placeholders

      return placeholders
