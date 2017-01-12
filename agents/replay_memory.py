import numpy as np
import collections


class ReplayMemory:
  def __init__(self, config):
    """
    capacity: total number of steps to store
    surrounding_steps: number of past and future steps to return - should this be passed to sample_batch???
    """
    self._capacity = config.replay_capacity
    self._surrounding_steps = config.optimality_tightening_steps
    self._discount_rate = config.discount_rate

    self._cursor = 0
    self._count = 0
    self._memory = [None for _ in range(self._capacity)]

  def start_new_episode(self, initial_observation):
    self.current_episode = EpisodeMemory(
        initial_observation, self._discount_rate, self._surrounding_steps)

  def store(self, observation, action, reward, done, next_observation):
    transition = self.current_episode.add_step(action, reward, done,
                                               next_observation)

    self._memory[self._cursor] = transition
    self._cursor = (self._cursor + 1) % self._capacity
    self._count = min(self._count + 1, self._capacity)

  def sample_batch(self, batch_size):
    observations = []
    actions = []
    rewards = []
    dones = []
    next_observations = []
    past_observations = []
    past_actions = []
    past_rewards = []
    past_discounts = []
    future_observations = []
    future_dones = []
    future_rewards = []
    future_discounts = []
    total_rewards = []

    for i in range(batch_size):
      random_index = np.random.randint(self._count)
      transition = self._memory[random_index]

      observations.append(transition.observation)
      actions.append(transition.action)
      rewards.append(transition.reward)
      dones.append(transition.done)
      next_observations.append(transition.next_observation)
      past_observations.append(transition.past_observations)
      past_actions.append(transition.past_actions)
      past_rewards.append(transition.past_rewards)
      past_discounts.append(transition.past_discounts)
      future_observations.append(transition.future_observations)
      future_dones.append(transition.future_dones)
      future_rewards.append(transition.future_rewards)
      future_discounts.append(transition.future_discounts)
      total_rewards.append(transition.total_reward)

    observations = np.stack(observations, axis=0)
    actions = np.array(actions)
    rewards = np.array(rewards)
    dones = np.array(dones)
    next_observations = np.stack(next_observations, axis=0)
    past_observations = np.stack(past_observations, axis=0)
    past_actions = np.stack(past_actions, axis=0)
    past_rewards = np.stack(past_rewards, axis=0)
    past_discounts = np.stack(past_discounts, axis=0)
    future_observations = np.stack(future_observations, axis=0)
    future_dones = np.stack(future_dones, axis=0)
    future_rewards = np.stack(future_rewards, axis=0)
    future_discounts = np.stack(future_discounts, axis=0)
    total_rewards = np.array(total_rewards)

    return SampleBatch(observations, actions, rewards, dones,
                       next_observations, past_observations, past_actions,
                       past_rewards, past_discounts, future_observations,
                       future_dones, future_rewards, future_discounts,
                       total_rewards)


SampleBatch = collections.namedtuple(
    'SampleBatch',
    ('observations', 'actions', 'rewards', 'dones', 'next_observations',
     'past_observations', 'past_actions', 'past_rewards', 'past_discounts',
     'future_observations', 'future_dones', 'future_rewards',
     'future_discounts', 'total_rewards'))


class EpisodeMemory:
  def __init__(self, initial_observation, discount_rate, surrounding_steps):
    self.surrounding_steps = surrounding_steps
    self.discount_rate = discount_rate

    self.observations = ExtendedList(initial_observation)
    self.actions = ExtendedList()
    self.rewards = ExtendedList()
    self.total_rewards = ExtendedList()
    self.timestep = -1
    self.final_timestep = None

  def add_step(self, action, reward, done, next_observation):
    self.timestep += 1
    self.actions.append(action)
    self.rewards.append(reward)
    self.observations.append(next_observation)
    self.total_rewards.append(reward)

    if done:
      self.compute_total_rewards()
      self.final_timestep = self.timestep

    return Transition(self, self.timestep, self.surrounding_steps)

  def compute_total_rewards(self):
    reward = 0
    for i in range(len(self.rewards) - 1, -1, -1):
      reward = reward * self.discount_rate + self.rewards[i]
      self.total_rewards[i] = reward


class Transition:
  def __init__(self, episode, timestep, surrounding_steps):
    self.episode = episode
    self.timestep = timestep
    self.surrounding_steps = surrounding_steps

  @property
  def observation(self):
    return self.episode.observations[self.timestep]

  @property
  def action(self):
    return self.episode.actions[self.timestep]

  @property
  def reward(self):
    return self.episode.rewards[self.timestep]

  @property
  def done(self):
    if self.episode.final_timestep != None:
      return self.timestep >= self.episode.final_timestep
    else:
      return False

  @property
  def next_observation(self):
    return self.episode.observations[self.timestep + 1]

  @property
  def past_observations(self):
    start = self.timestep - self.surrounding_steps
    end = self.timestep
    return self.episode.observations[start:end]

  @property
  def past_actions(self):
    start = self.timestep - self.surrounding_steps
    end = self.timestep
    return self.episode.actions[start:end]

  @property
  def past_rewards(self):
    past_rewards = []
    discount = 1
    reward = 0

    start = self.timestep - 1
    for i in range(start, start - self.surrounding_steps, -1):
      if i >= 0:
        discount /= self.episode.discount_rate
        reward += discount * self.episode.rewards[i]
      past_rewards.insert(0, reward)

    return past_rewards

  @property
  def past_discounts(self):
    past_discounts = []
    discount = 1

    start = self.timestep - 1
    for i in range(start, start - self.surrounding_steps, -1):
      if i >= 0:
        discount /= self.episode.discount_rate
      past_discounts.insert(0, discount)

    return past_discounts

  @property
  def future_observations(self):
    start = self.timestep + 2
    end = self.timestep + self.surrounding_steps + 2
    return self.episode.observations[start:end]

  @property
  def future_dones(self):
    if self.episode.final_timestep:
      start = self.timestep + 1
      end = self.timestep + self.surrounding_steps + 1
      return [
          timestep >= self.episode.final_timestep
          for timestep in range(start, end)
      ]
    else:
      return [False] * self.surrounding_steps

  @property
  def future_rewards(self):
    future_rewards = []
    discount = 1
    reward = self.reward

    start = self.timestep + 1
    for i in range(start, start + self.surrounding_steps):
      if i <= self.episode.timestep:
        discount *= self.episode.discount_rate
        reward += discount * self.episode.rewards[i]
      future_rewards.append(reward)

    return future_rewards

  @property
  def future_discounts(self):
    future_discounts = []
    discount = 1

    start = self.timestep + 1
    for i in range(start, start + self.surrounding_steps):
      if i <= self.episode.timestep:
        discount *= self.episode.discount_rate
      future_discounts.append(discount)

    return future_discounts

  @property
  def total_reward(self):
    return self.episode.total_rewards[self.timestep]


class ExtendedList(collections.MutableSequence):
  """Acts just like a list except for accessing indices outside range(0, len-1)

  For negative indices it returns the first item
  For indices >= len it returns the final item
  """

  def __init__(self, *items):
    self._list = [*items]

  def __delitem__(self, index):
    del self._list[index]

  def __getitem__(self, index):
    if isinstance(index, int):
      index = max(0, min(index, len(self._list) - 1))
      return self._list[index]
    elif isinstance(index, slice):
      start = index.start or 0
      stop = index.stop if index.stop != None else len(self)
      step = index.step or 1
      return [self[i] for i in range(start, stop, step)]
    else:
      return self._list[index]

  def __len__(self):
    return len(self._list)

  def __setitem__(self, index, item):
    self._list[index] = item

  def insert(self, index, item):
    self._list.insert(index, item)
