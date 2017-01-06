import numpy as np
import tensorflow as tf
import scipy.misc
import gym
import time
from datetime import datetime

from . import dqn
from .replay_memory import ReplayMemory

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('env_name', 'SpaceInvaders-v0',
                           'The OpenAI Gym environment to train on')
tf.app.flags.DEFINE_integer('num_episodes', 100,
                            'Number of episodes to train on')
tf.app.flags.DEFINE_string('train_dir', '/tmp/train_dqn',
                           'Directory to write checkpoints')
tf.app.flags.DEFINE_integer(
    'summary_step_frequency', 100,
    'How many training steps between writing summaries')
tf.app.flags.DEFINE_bool('render', False, 'Show game during training')
tf.app.flags.DEFINE_integer('batch_size', 32, 'Batch size')
tf.app.flags.DEFINE_integer('replay_capacity', 100000, 'Size of replay memory')
tf.app.flags.DEFINE_integer(
    'replay_start_size', 50000,
    'Pre-populate the replay memory with this number of random actions')
tf.app.flags.DEFINE_integer(
    'target_network_update_frequency', 10000,
    'The number of parameter updates before the target network is updated')
tf.app.flags.DEFINE_integer('num_frames_per_action', 4,
                            'Number of frames to repeat the chosen action for')
tf.app.flags.DEFINE_float(
    'initial_exploration', 1.0,
    'Initial value of epsilon is epsilon-greedy exploration')
tf.app.flags.DEFINE_float(
    'final_exploration', 0.1,
    'Final value of epsilon is epsilon-greedy exploration')
tf.app.flags.DEFINE_integer(
    'final_exploration_frame', 1000000,
    'The number of frames over to anneal epsilon to its final value')


def train(config):
  # Create environment
  env = gym.make(config.env_name)
  config.num_actions = env.action_space.n
  global_step = tf.contrib.framework.get_or_create_global_step()

  # Initialize replay memory
  replay_memory = ReplayMemory(capacity=config.replay_capacity)
  prepopulate_replay_memory(replay_memory, env, config)

  # Create action-value network
  with tf.variable_scope('action_value') as action_value_scope:
    input_frames, action_values, _, max_action = \
                                dqn.deep_q_network(config, env.action_space.n)

  # Create target action-value network
  with tf.variable_scope('target_action_value') as target_action_value_scope:
    target_input_frames, _, target_max_values, _ = dqn.deep_q_network(
        config, env.action_space.n)

  # Calculate Q loss
  reward_input, action_input, done_input, loss = dqn.loss(
      target_max_values, action_values, config)

  # Create training operation
  train_op = dqn.train(loss, global_step)

  # Create operation to copy action-values variables to target network
  reset_target_network = dqn.copy_q_network(
      action_value_scope, target_action_value_scope, 'reset_target_network')

  # Summary operation
  summary_op = tf.summary.merge_all()
  summary_writer = tf.summary.FileWriter(config.train_dir)

  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=config.train_dir,
      save_summaries_steps=0  # Summaries will be saved with train_op only
  ) as session:

    # Initialize
    session.run(reset_target_network)
    step = 0

    for episode in range(config.num_episodes):
      # Start episode with random action
      start_time = time.time()
      observation, reward, done, last_frame = repeat_action(
          env,
          action=env.action_space.sample(),
          k=config.num_frames_per_action,
          last_frame=env.reset(),
          render=config.render,
          config=config)
      episode_reward = reward
      episode_steps = 0

      # Play until losing
      while not done:
        step += 1
        episode_steps += 1

        # Epsilon greedy exploration/exploitation
        if np.random.rand() < epsilon(step, config):
          action = env.action_space.sample()
        else:
          action = session.run(max_action, {input_frames: [observation]})

        # Take action
        next_observation, reward, done, last_frame = repeat_action(
            env,
            action,
            k=config.num_frames_per_action,
            last_frame=last_frame,
            render=config.render,
            config=config)
        replay_memory.store(observation, action, reward, done,
                            next_observation)
        observation = next_observation
        episode_reward += reward

        # Train on random batch
        observations, actions, rewards, dones, next_observations = \
                                  replay_memory.sample_batch(config.batch_size)
        feed_dict = {
            input_frames: observations,
            target_input_frames: next_observations,
            reward_input: rewards,
            action_input: actions,
            done_input: dones
        }
        if step % config.summary_step_frequency == 0:
          # Don't write summaries every step
          _, summary = session.run([train_op, summary_op], feed_dict)
          summary_writer.add_summary(summary, step)
        else:
          session.run(train_op, feed_dict)

        # Reset target_action_value network
        if step % config.target_network_update_frequency == 0:
          reset_target_network(action_value_scope, target_action_value_scope)

      # Log episode
      log_episode(episode, start_time, episode_reward, episode_steps)


def prepopulate_replay_memory(replay_memory, env, config):
  """Play game with random actions to populate the replay memory"""

  done = True
  count = 0

  while count < config.replay_start_size:
    if done:
      last_frame = env.reset()
      observation = None

    action = env.action_space.sample()
    next_observation, reward, done, last_frame = repeat_action(
        env,
        action,
        k=config.num_frames_per_action,
        last_frame=last_frame,
        render=False,
        config=config)

    # Only store once there are 2 observations for the episode
    if observation:
      replay_memory.store(observation, action, reward, done, next_observation)
      count += 1

    observation = next_observation


def repeat_action(env, action, k, last_frame, render, config):
  """Repeat action for k steps and accumulate results"""

  observations = []
  reward = 0
  done = False

  for i in range(k):
    if render: env.render()

    frame, reward_, done_, _ = env.step(action)

    observations.append(preprocess(last_frame, frame, config))
    last_frame = frame
    reward += reward_
    done = done or done_

  return observations, reward, done, last_frame


def preprocess(previous_frame, frame, config):
  """Merge 2 frames, resize, and extract luminance"""

  # Max the 2 frames
  frames = np.stack([previous_frame, frame], axis=3)
  frame = frames.max(axis=3)

  # Rescale image
  frame = scipy.misc.imresize(frame, (config.input_height, config.input_width))

  # Calculate luminance
  luminance_ratios = [0.2126, 0.7152, 0.0722]
  frame = luminance_ratios * frame.astype(float)
  frame = frame.sum(axis=2)

  # Normalize each pixel between 0 and 1
  return frame / 255.0


def epsilon(step, config):
  """Epsilon is linearly annealed from an initial exploration value
  to a final exploration value over a number of steps"""

  annealing_rate = (config.initial_exploration - config.final_exploration
                    ) / config.final_exploration_frame
  annealed_exploration = config.initial_exploration - (step * annealing_rate)
  return max(annealed_exploration, config.final_exploration)


def log_episode(episode, start_time, reward, steps):
  now = datetime.strftime(datetime.now(), '%x %X')
  duration = time.time() - start_time
  steps_per_sec = steps / duration
  format_string = ('%s: Episode = %d, reward = %.0f '
                   '(steps = %d, secs = %.2f, %.2f steps/sec)')
  print(format_string % (now, episode, reward, steps, duration, steps_per_sec))


def main(_):
  train(FLAGS)


if __name__ == '__main__':
  tf.app.run()
