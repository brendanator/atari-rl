from datetime import datetime
import os
import time

import tensorflow as tf
import numpy as np

from agents.agent import Agent

flags = tf.app.flags

# Environment
flags.DEFINE_string('game', 'SpaceInvaders',
                    'The Arcade Learning Environment game to play')
flags.DEFINE_string('frameskip', '4',
                    'Number of frames to repeat actions for. '
                    'Can be int or tuple with min and max+1')
flags.DEFINE_float(
    'repeat_action_probability', 0.25,
    'Probability of ignoring the agent action and repeat last action')
flags.DEFINE_string('input_shape', '[84, 84]', 'Rescale input to this shape')
flags.DEFINE_integer('input_frames', 4, 'Number of frames to input')
flags.DEFINE_integer('random_reset_actions', 30,
                     'Number of random actions to perform at start of episode')

# Agent
flags.DEFINE_bool('double_q', False, 'Whether to use Double Q-Learning')
flags.DEFINE_bool('bootstrapped', False, 'Whether to use bootstrapped DQN')
flags.DEFINE_integer('num_bootstrap_heads', 10,
                     'Number of bootstrapped head to use')
flags.DEFINE_float('bootstrap_mask_probability', 1.0,
                   'Probability each head has of training on each experience')
flags.DEFINE_bool(
    'bootstrap_use_ensemble', False,
    'Choose action with most bootstrap heads votes. Use only in evaluation')
flags.DEFINE_integer('replay_capacity', 100000, 'Size of replay memory')
flags.DEFINE_integer(
    'replay_start_size', 50000,
    'Pre-populate the replay memory with this number of random actions')
flags.DEFINE_bool('replay_prioritized', False,
                  'Enable prioritized replay memory')
flags.DEFINE_float('replay_alpha', 0.6,
                   'Prioritized experience replay exponent')
flags.DEFINE_float('replay_beta', 0.4, 'Initial importance sampling exponent')
flags.DEFINE_bool('persistent_advantage_learning', False,
                  'Enable persistent advantage learning')
flags.DEFINE_float('pal_alpha', 0.9, 'Persistent advantage learning alpha')
flags.DEFINE_integer('train_period', 4,
                     'The number of steps between training updates')
flags.DEFINE_integer(
    'target_network_update_period', 10000,
    'The number of parameter updates before the target network is updated')
flags.DEFINE_bool('dueling', False, 'Enable dueling network architecture')
flags.DEFINE_bool('optimality_tightening', False,
                  'Enable optimality tightening')
flags.DEFINE_integer(
    'optimality_tightening_steps', 4,
    'How many steps to use for calculate bounds in optimality tightening')
flags.DEFINE_float(
    'optimality_penalty_ratio', 4.0,
    'The ratio of constraint violation penalty compared to target loss')
flags.DEFINE_bool('exploration_bonus', False,
                  'Enable pseudo-count based exploration bonus')
flags.DEFINE_float('exploration_beta', 0.05,
                   'Value to scale the exploration bonus by')
flags.DEFINE_string('exploration_image_shape', '[42, 42]',
                    'Shape of image to use with CTS in exploration bonus')

# Training
flags.DEFINE_integer('batch_size', 32, 'Batch size')
flags.DEFINE_integer('num_steps', 50000000, 'Number of steps to train on')
flags.DEFINE_string('train_dir', 'checkpoints',
                    'Directory to write checkpoints')
flags.DEFINE_integer('summary_step_period', 100,
                     'How many training steps between writing summaries')
flags.DEFINE_float('discount_rate', 0.99, 'Discount rate for future rewards')
flags.DEFINE_float('initial_exploration', 1.0,
                   'Initial value of epsilon is epsilon-greedy exploration')
flags.DEFINE_float('final_exploration', 0.1,
                   'Final value of epsilon is epsilon-greedy exploration')
flags.DEFINE_integer(
    'final_exploration_frame', 1000000,
    'The number of frames over to anneal epsilon to its final value')
flags.DEFINE_float('reward_clipping', 1.0,
                   'Range around zero to limit rewards to. 0 to disable')
flags.DEFINE_float('loss_clipping', 1.0,
                   'Range around zero to limit loss to. 0 to disable')
flags.DEFINE_float('grad_clipping', 10.0,
                   'Range around zero to limit gradients to. 0 to disable')

# Render
flags.DEFINE_bool('render', False, 'Show game during training')


def train(config):
  # Create agent
  agent = Agent(config)

  # Summary writer
  checkpoint_dir = os.path.join(config.train_dir, config.game)
  summary_writer = tf.summary.FileWriter(checkpoint_dir)

  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=checkpoint_dir,
      save_summaries_steps=0  # Summaries will be saved with train_op only
  ) as session:

    # Initialize counts
    step, episode = 0, 0

    while step < config.num_steps:
      # Start new episode
      start_time = time.time()
      observation, episode_score, done = agent.new_game()
      episode += 1
      episode_steps = 0

      # Play until losing
      while not done:
        # Reset target action-value network
        if step % config.target_network_update_period == 0:
          agent.reset_target_network(session)

        # Increment counts
        step += 1
        episode_steps += 1

        # Choose next action
        action = agent.action(observation, step, session)

        # Take action
        observation, reward, done = agent.take_action(action)
        episode_score += reward

        # Train on random batch
        if step % config.train_period == 0:
          if step % config.summary_step_period == 0:
            summary = agent.train(session, step, summary=True)
            summary_writer.add_summary(summary, step)
          else:
            agent.train(session, step)

      # Log episode
      log_episode(episode, start_time, episode_score, episode_steps)


def log_episode(episode, start_time, score, steps):
  now = datetime.strftime(datetime.now(), '%F %X')
  duration = time.time() - start_time
  steps_per_sec = steps / duration
  format_string = ('%s: Episode %d, score %.0f '
                   '(%d steps, %.2f secs, %.2f steps/sec)')
  print(format_string % (now, episode, score, steps, duration, steps_per_sec))


def main(_):
  config = flags.FLAGS
  config.frameskip = eval(config.frameskip)
  config.input_shape = eval(config.input_shape)
  config.exploration_image_shape = eval(config.exploration_image_shape)

  train(config)


if __name__ == '__main__':
  tf.app.run()
