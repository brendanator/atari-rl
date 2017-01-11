import tensorflow as tf
import numpy as np

import time
from datetime import datetime

from atari.atari import Atari
from agents import dqn
from agents.replay_memory import ReplayMemory

flags = tf.app.flags

# Environment
flags.DEFINE_string('game', 'SpaceInvaders-v0',
                    'The OpenAI Gym environment to train on')
flags.DEFINE_integer('input_height', 84, 'Rescale input to this height')
flags.DEFINE_integer('input_width', 84, 'Rescale input to this width')
flags.DEFINE_integer('input_frames', 4, 'Number of frames to input')
flags.DEFINE_integer('num_frames_per_action', 4,
                     'Number of frames to repeat the chosen action for')
flags.DEFINE_integer('random_reset_actions', 30,
                     'Number of random actions to perform at start of episode')

# Agent
flags.DEFINE_bool('double_q', False, 'Whether to use double Q-Learning')
flags.DEFINE_integer('replay_capacity', 100000, 'Size of replay memory')
flags.DEFINE_integer(
    'replay_start_size', 50000,
    'Pre-populate the replay memory with this number of random actions')
flags.DEFINE_integer(
    'target_network_update_frequency', 10000,
    'The number of parameter updates before the target network is updated')

# Training
flags.DEFINE_integer('batch_size', 32, 'Batch size')
flags.DEFINE_integer('num_episodes', 100, 'Number of episodes to train on')
flags.DEFINE_string('train_dir', '/tmp/train_atari',
                    'Directory to write checkpoints')
flags.DEFINE_integer('summary_step_frequency', 100,
                     'How many training steps between writing summaries')
flags.DEFINE_float('discount_factor', 0.99,
                   'Discount factor for future rewards')
flags.DEFINE_float('initial_exploration', 1.0,
                   'Initial value of epsilon is epsilon-greedy exploration')
flags.DEFINE_float('final_exploration', 0.1,
                   'Final value of epsilon is epsilon-greedy exploration')
flags.DEFINE_integer(
    'final_exploration_frame', 1000000,
    'The number of frames over to anneal epsilon to its final value')

# Render
flags.DEFINE_bool('render', False, 'Show game during training')

FLAGS = flags.FLAGS


def train(config):
  # Create environment
  atari = Atari(config.game, config)
  global_step = tf.contrib.framework.get_or_create_global_step()

  # Initialize replay memory
  replay_memory = ReplayMemory(capacity=config.replay_capacity)
  prepopulate_replay_memory(replay_memory, atari, config.replay_start_size)

  # Create action-value network
  policy_network = dqn.PolicyNetwork(config)

  # Create target action-value network
  target_network = dqn.TargetNetwork(config)

  # Calculate Q loss
  action_input, reward_input, done_input, loss = dqn.loss(
      policy_network, target_network, config)

  # Create training operation
  train_op = dqn.train(loss, global_step)

  # Create operation to copy variables from policy network to target network
  reset_target_network = policy_network.copy_to_network(target_network)

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
      observation, episode_reward = atari.reset()
      done = False
      episode_steps = 0

      # Play until losing
      while not done:
        step += 1
        episode_steps += 1

        # Epsilon greedy exploration/exploitation
        if np.random.rand() < epsilon(step, config):
          action = atari.sample_action()
        else:
          action = session.run(policy_network.max_action,
                               {policy_network.input_frames: [observation]})

        # Take action
        next_observation, reward, done = atari.step(action)
        replay_memory.store(observation, action, reward, done,
                            next_observation)
        observation = next_observation
        episode_reward += reward

        # Train on random batch
        observations, actions, rewards, dones, next_observations = \
                                  replay_memory.sample_batch(config.batch_size)
        feed_dict = {
            policy_network.input_frames: observations,
            action_input: actions,
            reward_input: rewards,
            done_input: dones,
            target_network.input_frames: next_observations,
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


def prepopulate_replay_memory(replay_memory, atari, start_size):
  """Play game with random actions to populate the replay memory"""

  done = True

  for _ in range(start_size):
    if done: observation, _ = atari.reset(render=False)

    action = atari.sample_action()
    next_observation, reward, done = atari.step(action, render=False)
    replay_memory.store(observation, action, reward, done, next_observation)
    observation = next_observation


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
  format_string = ('%s: Episode %d, reward %.0f '
                   '(%d steps, %.2f secs, %.2f steps/sec)')
  print(format_string % (now, episode, reward, steps, duration, steps_per_sec))


def main(_):
  train(FLAGS)


if __name__ == '__main__':
  tf.app.run()
