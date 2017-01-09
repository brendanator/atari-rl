import gym
import numpy as np
import scipy.misc


class Atari:
  def __init__(self, game, config):
    self.env = gym.make(config.game)

    self.random_reset_actions = config.random_reset_actions
    self.input_height = config.input_height
    self.input_width = config.input_width
    self.num_frames_per_action = config.num_frames_per_action
    self.render = config.render

    config.num_actions = self.num_actions

  @property
  def num_actions(self):
    return self.env.action_space.n

  def sample_action(self):
    return self.env.action_space.sample()

  def reset(self, render=None):
    """Reset the game and play some random actions"""

    if render == None: render = self.render

    self._previous_frame = self.env.reset()
    reward = 0
    observation = []

    for i in range(self.random_reset_actions):
      if render: env.render()

      frame, reward_, done, _ = self.env.step(self.sample_action())

      observation.append(self._process(frame))
      reward += reward_
      if done: self.reset(render)

    return np.array(observation[-4:]), reward

  def step(self, action, render=None):
    """Repeat action for k steps and accumulate results"""

    if render == None: render = self.render

    observation = []
    reward = 0
    done = False

    for i in range(self.num_frames_per_action):
      if render: env.render()

      frame, reward_, done_, _ = self.env.step(action)

      observation.append(self._process(frame))
      reward += reward_
      done = done or done_

    return np.array(observation), reward, done

  def _process(self, frame):
    """Merge 2 frames, resize, and extract luminance"""

    # Max the 2 frames
    frames = np.stack([self._previous_frame, frame], axis=3)
    frame = frames.max(axis=3)
    self._previous_frame = frame

    # Rescale image
    frame = scipy.misc.imresize(frame, (self.input_height, self.input_width))

    # Calculate luminance
    luminance_ratios = [0.2126, 0.7152, 0.0722]
    frame = luminance_ratios * frame.astype(float)
    frame = frame.sum(axis=2)

    # Normalize each pixel between 0 and 1
    return frame / 255.0
