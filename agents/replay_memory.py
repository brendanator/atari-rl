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

    # Track position and count in memory
    self.cursor = 0
    self.count = 0

    # Store all value in numpy arrays
    self.observations = np.zeros(
        [
            config.replay_capacity, config.input_frames, config.input_height,
            config.input_width
        ],
        dtype=np.float32)
    self.actions = np.zeros([config.replay_capacity], dtype=np.uint8)
    self.rewards = np.zeros([config.replay_capacity], dtype=np.float32)
    self.total_rewards = np.zeros([config.replay_capacity], dtype=np.float32)
    # Store alive instead of done as it simplifies calculations elsewhere
    self.alives = np.zeros([config.replay_capacity], dtype=np.bool)

  def store(self, observation, action, reward, done):
    self.observations[self.cursor] = observation
    self.actions[self.cursor] = action
    self.rewards[self.cursor] = reward
    self.total_rewards[self.cursor] = reward
    self.alives[self.cursor] = not done

    if done:
      # Update total_rewards for episode
      i = self.cursor - 1
      while self.alives[i] and i < self.count:
        reward = reward * self.discount_rate + self.rewards[i]
        self.total_rewards[i] = reward
        i = (i - 1) % self.capacity

    self.cursor = (self.cursor + 1) % self.capacity
    self.count = min(self.count + 1, self.capacity)

  def sample_batch(self, batch_size):
    indices = self.sample_indices(batch_size)

    observations = self.observations[indices]
    actions = self.actions[indices]
    rewards = self.rewards[indices]
    alives = self.alives[indices]
    next_observations = self.observations[(indices + 1) % self.capacity]
    total_rewards = self.total_rewards[indices]

    indices = indices.reshape(-1, 1)
    past_offsets = np.arange(-1, -self.constraint_steps - 1, -1)
    past_indices = (indices + past_offsets) % self.capacity
    past_observations = self.observations[past_indices]
    past_actions = self.actions[past_indices]
    past_rewards = self.rewards[past_indices]
    past_alives = self.alives[past_indices]

    future_offsets = np.arange(1, self.constraint_steps + 1)
    future_indices = (indices + future_offsets) % self.capacity
    future_observations = self.observations[(future_indices + 1) %
                                            self.capacity]
    future_rewards = self.rewards[future_indices]
    future_alives = self.alives[future_indices]

    return SampleBatch(observations, actions, rewards, alives,
                       next_observations, past_observations, past_actions,
                       past_rewards, past_alives, future_observations,
                       future_rewards, future_alives, total_rewards)

  def sample_indices(self, batch_size):
    indices = []

    def valid_index(index):
      # Indices must be unique
      if index in indices:
        return False

      # Don't return states that may have incomplete constraint data
      offset = self.constraint_steps + 1
      close_below = (index <= self.cursor) and (self.cursor <= index + offset)
      close_above = (index - offset <= self.cursor) and (self.cursor <= index)
      if close_below or close_above:
        return False

      return True

    for i in range(batch_size):
      index = np.random.randint(self.count)
      while not valid_index(index):
        index = np.random.randint(self.count)

      indices.append(index)

    return np.array(indices)


SampleBatch = collections.namedtuple('SampleBatch', (
    'observations', 'actions', 'rewards', 'alives', 'next_observations',
    'past_observations', 'past_actions', 'past_rewards', 'past_alives',
    'future_observations', 'future_rewards', 'future_alives', 'total_rewards'))
