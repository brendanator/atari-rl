import numpy as np
import re

from gym.envs.atari.atari_env import AtariEnv


class Atari(object):
  def __init__(self, config):
    game = '_'.join(
        [g.lower() for g in re.findall('[A-Z]?[a-z]+', config.game)])

    print('Playing %s with frameskip %s and repeat_action_probability %s' %
          (game, str(config.frameskip), str(config.repeat_action_probability)))

    self.env = AtariEnv(
        game=game,
        obs_type='image',
        frameskip=config.frameskip,
        repeat_action_probability=config.repeat_action_probability)

    if isinstance(config.frameskip, int):
      frameskip = config.frameskip
    else:
      frameskip = config.frameskip[1]

    self.input_frames = config.input_frames
    self.max_noops = config.max_noops / frameskip
    self.render = config.render

    config.num_actions = self.env.action_space.n

  def sample_action(self):
    return self.env.action_space.sample()

  def reset(self):
    """Reset the game and play some random actions"""

    frame = self.env.reset()
    if self.render: env.render()
    self.frames = [frame]
    reward = 0

    for i in range(np.random.randint(self.input_frames, self.max_noops + 1)):
      frame, reward_, done, _ = self.env.step(0)
      if self.render: env.render()

      self.frames.append(frame)
      reward += reward_
      if done: self.reset(render)

    return self.frames, reward, done

  def step(self, action):
    frame, reward, done, _ = self.env.step(action)
    if self.render: env.render()
    self.frames.append(frame)
    return self.frames, reward, done
