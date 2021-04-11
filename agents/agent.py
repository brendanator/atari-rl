import numpy as np

from atari import Atari
from agents.exploration_bonus import ExplorationBonus


class Agent(object):
  def __init__(self, policy_network, replay_memory, summary, config):
    self.config = config
    self.policy_network = policy_network
    self.replay_memory = replay_memory
    self.summary = summary

    # Create environment
    self.atari = Atari(summary, config)
    self.exploration_bonus = ExplorationBonus(config)

  def new_game(self):
    self.policy_network.sample_head()
    observation, reward, done = self.atari.reset()
    self.replay_memory.store_new_episode(observation)
    return observation, reward, done

  def action(self, session, step, observation):
    # Epsilon greedy exploration/exploitation even for bootstrapped DQN
    if np.random.rand() < self.epsilon(step):
      return self.atari.sample_action()
    [action] = session.run(
        self.policy_network.choose_action,
        {self.policy_network.inputs.observations: [observation]})
    return action

  def epsilon(self, step):
    """Epsilon is linearly annealed from an initial exploration value
    to a final exploration value over a number of steps"""

    initial = self.config.initial_exploration
    final = self.config.final_exploration
    final_frame = self.config.final_exploration_frame

    annealing_rate = (initial - final) / final_frame
    annealed_exploration = initial - (step * annealing_rate)
    epsilon = max(annealed_exploration, final)

    self.summary.epsilon(step, epsilon)

    return epsilon

  def take_action(self, action):
    observation, reward, done = self.atari.step(action)
    training_reward = self.process_reward(reward, observation)

    # Store action, reward and done with the next observation
    self.replay_memory.store_transition(action, training_reward, done,
                                        observation)

    return observation, reward, done

  def process_reward(self, reward, frames):
    if self.config.exploration_bonus:
      reward += self.exploration_bonus.bonus(frames)

    if self.config.reward_clipping:
      reward = max(-self.config.reward_clipping,
                   min(reward, self.config.reward_clipping))

    return reward

  def populate_replay_memory(self):
    """Play game with random actions to populate the replay memory"""

    count = 0
    done = True

    while count < self.config.replay_start_size or not done:
      if done: self.new_game()
      _, _, done = self.take_action(self.atari.sample_action())
      count += 1

    self.atari.episode = 0

  def log_episode(self, step):
    self.atari.log_episode(step)
