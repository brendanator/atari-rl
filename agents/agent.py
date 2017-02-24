import numpy as np

from atari import Atari
from agents.exploration_bonus import ExplorationBonus
import util


class Agent(object):
  def __init__(self, policy_network, replay_memory, config):
    self.config = config
    self.policy_network = policy_network
    self.replay_memory = replay_memory

    # Create environment
    self.atari = Atari(config)
    self.exploration_bonus = ExplorationBonus(config)

  def new_game(self):
    self.policy_network.sample_head()
    frames, reward, done = self.atari.reset()
    observation = self.process_frames(frames)
    self.replay_memory.store_new_episode(observation)
    return observation, reward, done

  def action(self, session, step, observation):
    # Epsilon greedy exploration/exploitation even for bootstrapped DQN
    if np.random.rand() < self.epsilon(step):
      return self.atari.sample_action()
    else:
      return session.run(self.policy_network.choose_action,
                         {self.policy_network.inputs.frames: [observation]})

  def epsilon(self, step):
    """Epsilon is linearly annealed from an initial exploration value
    to a final exploration value over a number of steps"""

    initial = self.config.initial_exploration
    final = self.config.final_exploration
    final_frame = self.config.final_exploration_frame

    annealing_rate = (initial - final) / final_frame
    annealed_exploration = initial - (step * annealing_rate)
    return max(annealed_exploration, final)

  def take_action(self, action):
    frames, reward, done = self.atari.step(action)
    training_reward = self.process_reward(reward, frames)
    observation = self.process_frames(frames)

    # Store action, reward and done with the next observation
    self.replay_memory.store_transition(action, training_reward, done,
                                        observation)

    return observation, reward, done

  def process_frames(self, frames):
    observation = []
    for i in range(-self.config.input_frames, 0):
      image = util.process_image(frames[i - 1], frames[i],
                                 self.config.input_shape)
      image /= 255.0  # Normalize each pixel between 0 and 1
      observation.append(image)
    return observation

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

  def log_episode(self):
    self.atari.log_episode()
