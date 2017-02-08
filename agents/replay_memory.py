import numpy as np
import collections


class ReplayMemory:
  def __init__(self, config):
    # Config
    self.capacity = config.replay_capacity
    self.discount_rate = config.discount_rate
    if config.optimality_tightening:
      self.constraint_steps = config.optimality_tightening_steps
    else:
      self.constraint_steps = 0
    self.prioritized = config.replay_prioritized

    # Track position and count in memory
    self.cursor = 0
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

    if self.prioritized:
      self.priorities = ProportionalPriorities(
          config.replay_capacity, config.replay_alpha, config.replay_beta,
          config.num_steps)

  def store(self, observation, action, reward, done):
    self.observations[self.cursor] = observation
    self.actions[self.cursor] = action
    self.rewards[self.cursor] = reward
    self.total_rewards[self.cursor] = reward
    self.alives[self.cursor] = not done
    if self.prioritized:
      self.priorities.set_to_max(self.cursor)

    if done:
      # Update total_rewards for episode
      i = self.cursor - 1
      while self.alives[i] and i < self.count:
        reward = reward * self.discount_rate + self.rewards[i]
        self.total_rewards[i] = reward
        i = (i - 1) % self.capacity

    self.cursor = (self.cursor + 1) % self.capacity
    self.count = min(self.count + 1, self.capacity)

  def update_priorities(self, indices, errors):
    errors = np.absolute(errors)
    if self.prioritized:
      for index, error in zip(indices, errors):
        self.priorities.update(index, error)

  def sample_batch(self, batch_size, step):
    indices = self.sample_indices(batch_size)

    observations = self.observations[indices]
    actions = self.actions[indices]
    rewards = self.rewards[indices]
    alives = self.alives[indices]
    next_observations = self.observations[(indices + 1) % self.capacity]

    if self.prioritized:
      error_weights = self.priorities.error_weights(indices, self.count, step)
    else:
      error_weights = np.ones_like(indices)

    past_offsets = np.arange(-1, -self.constraint_steps - 1, -1)
    past_indices = (indices.reshape(-1, 1) + past_offsets) % self.capacity
    past_observations = self.observations[past_indices]
    past_actions = self.actions[past_indices]
    past_rewards = self.rewards[past_indices]
    past_alives = self.alives[past_indices]

    future_offsets = np.arange(1, self.constraint_steps + 1)
    future_indices = (indices.reshape(-1, 1) + future_offsets) % self.capacity
    future_observations = self.observations[(future_indices + 1) %
                                            self.capacity]
    future_rewards = self.rewards[future_indices]
    future_alives = self.alives[future_indices]

    total_rewards = self.total_rewards[indices]
    bootstrap_mask = self.bootstrap_masks[indices]

    return SampleBatch(indices, observations, actions, rewards, alives,
                       next_observations, error_weights, past_observations,
                       past_actions, past_rewards, past_alives,
                       future_observations, future_rewards, future_alives,
                       total_rewards, bootstrap_mask)

  def sample_indices(self, batch_size):
    indices = []

    for i in range(batch_size):
      index = self.sample_index()
      while index in indices or not self.valid_index(index):
        index = self.sample_index()

      indices.append(index)

    return np.array(indices)

  def sample_index(self):
    if self.prioritized:
      return self.priorities.sample_index()
    else:
      return np.random.randint(self.count)

  def valid_index(self, index):
    # Don't return states that may have incomplete constraint data
    offset = self.constraint_steps + 1
    close_below = (index <= self.cursor) and (self.cursor <= index + offset)
    close_above = (index - offset <= self.cursor) and (self.cursor <= index)

    return not (close_below or close_above)


SampleBatch = collections.namedtuple(
    'SampleBatch',
    ('indices', 'observations', 'actions', 'rewards', 'alives',
     'next_observations', 'error_weights', 'past_observations', 'past_actions',
     'past_rewards', 'past_alives', 'future_observations', 'future_rewards',
     'future_alives', 'total_rewards', 'bootstrap_mask'))


class ProportionalPriorities:
  """Track the priorities of each transition proportional to the TD-error

  Contains a sum tree and a max tree for tracking values needed
  Each tree is implemented with an np.array for efficiency"""

  def __init__(self, capacity, alpha, beta, num_steps):
    self.capacity = capacity
    self.alpha = alpha
    self.beta = beta
    self.beta_grad = (1.0 - beta) / num_steps

    self.sum_tree = np.zeros(2 * capacity - 1, dtype=np.float)
    self.max_tree = np.zeros(2 * capacity - 1, dtype=np.float)

  def total_priority(self):
    return self.sum_tree[0]

  def max_priority(self):
    return self.max_tree[0] or 1  # Default priority if tree is empty

  def set_to_max(self, leaf_index):
    self.update_scaled(leaf_index, self.max_priority())

  def update(self, leaf_index, priority):
    scaled_priority = priority**self.alpha
    self.update_scaled(leaf_index, scaled_priority)

  def update_scaled(self, leaf_index, scaled_priority):
    index = leaf_index + (self.capacity - 1)  # Skip the sum nodes

    self.sum_tree[index] = scaled_priority
    self.max_tree[index] = scaled_priority

    self.update_parent_priorities(index)

  def update_parent_priorities(self, index):
    parent = self.parent(index)
    sibling = self.sibling(index)

    self.sum_tree[parent] = self.sum_tree[index] + self.sum_tree[sibling]
    self.max_tree[parent] = max(self.max_tree[index], self.max_tree[sibling])

    if parent > 0:
      self.update_parent_priorities(parent)

  def sample_index(self):
    sample_value = np.random.random() * self.total_priority()
    return self.index_of_value(sample_value)

  def index_of_value(self, value):
    index = 0
    while True:
      if self.is_leaf(index):
        return index - (self.capacity - 1)

      left_index = self.left_child(index)
      left_value = self.sum_tree[left_index]
      if value <= left_value:
        index = left_index
      else:
        index = self.right_child(index)
        value -= left_value

  def error_weights(self, indices, count, step):
    probabilities = self.sum_tree[indices + (self.capacity - 1)]
    beta = self.annealed_beta(step)
    error_weights = (1.0 / (count * probabilities))**beta
    return error_weights / error_weights.max()

  def annealed_beta(self, step):
    return self.beta + self.beta_grad * step

  def is_leaf(self, index):
    return index >= self.capacity - 1

  def parent(self, index):
    return (index - 1) // 2

  def sibling(self, index):
    if index % 2 == 0:
      return index - 1
    else:
      return index + 1

  def left_child(self, index):
    return (index * 2) + 1

  def right_child(self, index):
    return (index * 2) + 2

  def __str__(self):
    sum_tree, max_tree = '', ''
    index = 0
    while True:
      end_index = index * 2 + 1
      sum_tree += str(self.sum_tree[index:end_index]) + '\n'
      max_tree += str(self.max_tree[index:end_index]) + '\n'

      if index >= self.capacity: break
      index = end_index

    return ('Sum\n' + sum_tree + '\nMax\n' + max_tree).strip()
