import numpy as np
import tensorflow as tf

from .replay_priorities import ProportionalPriorities, UniformPriorities


class ReplayMemory(object):
  def __init__(self, input_offsets, config):
    # Input offsets that must be valid. Final offset can be safely ignored
    pre_offset = min(input_offsets)
    post_offset = max(input_offsets)
    self.input_range = np.arange(pre_offset, post_offset)

    # Config
    self.capacity = config.replay_capacity
    self.discount_rate = config.discount_rate
    self.recent_only = config.async is not None

    # Track position and count in memory
    self.cursor = -1  # Cursor points to index currently being written
    self.count = 0

    # Store all values in numpy arrays
    self.observations = np.zeros(
        [config.replay_capacity, config.input_frames] + config.input_shape,
        dtype=np.float32)
    self.actions = np.zeros([config.replay_capacity], dtype=np.uint8)
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

    # Avoid infinite loop from repeated collisions
    retries = 0
    max_retries = len(self.input_range) * batch_size

    index = self.offset_index(self.cursor, -max(self.input_range) - 1)
    for _ in range(batch_size):
      if self.valid_index(index, indices):
        indices.append(index)
      else:
        while retries < max_retries:
          retries += 1
          index = self.offset_index(index, -1)
          if self.valid_index(index, indices):
            indices.append(index)
            break
      index = self.offset_index(index, -len(self.input_range))

    return np.array(indices)

  def sample_indices(self, batch_size):
    indices = []

    # Avoid infinite loop from repeated collisions
    retries = 0
    max_retries = batch_size

    for _ in range(batch_size):
      index = self.priorities.sample_index(self.count)
      if self.valid_index(index, indices):
        indices.append(index)
      else:
        while retries < max_retries:
          retries += 1
          index = self.priorities.sample_index(self.count)
          if self.valid_index(index, indices):
            indices.append(index)
            break

    return np.array(indices)

  def valid_index(self, index, indices):
    # Reject indices too close to the cursor or to the start/end of episode
    index_range = self.offset_index(index, self.input_range)
    excludes_cursor = self.cursor not in index_range
    within_episode = self.alives[index_range].all()
    if not (excludes_cursor and within_episode):
      return False

    # Reject indices whose range overlaps existing indices ranges
    index_range = index_range.reshape([1, -1]).transpose()
    for other in indices:
      if (index_range == self.offset_index(other, self.input_range)).any():
        return False

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

  def update_priorities(self, errors):
    self.priorities.update_priorities(self.indices, errors)

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
