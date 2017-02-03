import gym


class Atari:
  def __init__(self, game, config):
    self.env = gym.make(config.game)

    self.random_reset_actions = config.random_reset_actions
    self.num_frames_per_action = config.num_frames_per_action
    self.render = config.render

    config.num_actions = self.env.action_space.n

  def sample_action(self):
    return self.env.action_space.sample()

  def reset(self):
    """Reset the game and play some random actions"""

    self.env.reset()
    reward = 0
    self.frames = []

    for i in range(self.random_reset_actions):
      if self.render: env.render()

      frame, reward_, done, _ = self.env.step(self.sample_action())

      self.frames.append(frame)
      reward += reward_
      if done: self.reset(render)

    return self.frames, reward, done

  def step(self, action):
    """Repeat action for k steps and accumulate results"""

    reward = 0
    done = False

    for i in range(self.num_frames_per_action):
      if self.render: env.render()

      frame, reward_, done_, _ = self.env.step(action)

      self.frames.append(frame)
      reward += reward_
      done = done or done_

    return self.frames, reward, done
