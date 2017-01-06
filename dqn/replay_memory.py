import numpy as np


class Transition:
  __slots__ = ('observation', 'action', 'reward', 'done', 'next_observation')

  def __init__(self, observation, action, reward, done, next_observation):
    self.observation = observation
    self.action = action
    self.reward = reward
    self.done = done
    self.next_observation = next_observation


class ReplayMemory:
  def __init__(self, capacity):
    self._capacity = capacity
    self._cursor = 0
    self._count = 0
    self._memory = [None for _ in range(capacity)]

  def store(self, observation, action, reward, done, next_observation):
    transition = Transition(observation, action, reward, done,
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

    for i in range(batch_size):
      random_index = np.random.randint(self._count)
      transition = self._memory[random_index]

      observations.append(transition.observation)
      actions.append(transition.action)
      rewards.append(transition.reward)
      dones.append(transition.done)
      next_observations.append(transition.next_observation)

    observations = np.stack(observations, axis=0)
    actions = np.array(actions)
    rewards = np.array(rewards)
    dones = np.array(dones)
    next_observations = np.stack(next_observations, axis=0)

    return observations, actions, rewards, dones, next_observations
