from datetime import datetime
import os
import time

import tensorflow as tf
import numpy as np

from atari.atari import Atari
from agents import dqn, util
from agents.optimality_tightening import ConstraintNetwork
from agents.replay_memory import ReplayMemory
from agents.exploration_bonus import ExplorationBonus

flags = tf.app.flags

# Environment
flags.DEFINE_string('game', 'SpaceInvaders-v0',
                    'The OpenAI Gym environment to train on')
flags.DEFINE_string('input_shape', '[84, 84]', 'Rescale input to this shape')
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
flags.DEFINE_bool('replay_prioritized', False,
                  'Enable prioritized replay memory')
flags.DEFINE_float('alpha', 0.6, 'Prioritized experience replay exponent')
flags.DEFINE_float('beta', 0.4, 'Initial importance sampling exponent')
flags.DEFINE_integer(
    'target_network_update_frequency', 10000,
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
flags.DEFINE_integer('summary_step_frequency', 100,
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
  # Create environment
  atari = Atari(config.game, config)
  global_step = tf.contrib.framework.get_or_create_global_step()
  exploration_bonus = ExplorationBonus(config)

  # Initialize replay memory
  replay_memory = ReplayMemory(config)
  prepopulate_replay_memory(replay_memory, atari, exploration_bonus, config)

  # Create action-value network
  policy_network = dqn.PolicyNetwork(config)

  # Create target action-value network
  target_network = dqn.TargetNetwork(config)

  # Calculate loss
  td_errors = target_network.square_error(policy_network)

  if config.optimality_tightening:
    constraint_network = ConstraintNetwork(config)
    violation_penalty, error_rescaling = constraint_network.violation_penalty(
        policy_network)
    td_errors = (td_errors + violation_penalty) / error_rescaling
  else:
    constraint_network = None

  error_weights = tf.placeholder(tf.float32, [None], 'error_weights')
  loss = tf.reduce_mean(error_weights * td_errors)

  if config.loss_clipping > 0:
    loss = tf.maximum(-config.loss_clipping,
                      tf.minimum(loss, config.loss_clipping))

  # Create training operation
  train_op = optimize(loss, global_step, config)

  # Create operation to copy variables from policy network to target network
  reset_target_network = policy_network.copy_to_network(target_network)

  # Summary operation
  summary_op = tf.summary.merge_all()
  checkpoint_dir = os.path.join(config.train_dir, config.game)
  summary_writer = tf.summary.FileWriter(checkpoint_dir)

  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=checkpoint_dir,
      save_summaries_steps=0  # Summaries will be saved with train_op only
  ) as session:

    # Initialize
    session.run(reset_target_network)
    step = 0
    episode = 0

    while step < config.num_steps:
      # Start episode with random action
      start_time = time.time()
      frames, episode_score, done = atari.reset()
      observation = process_frames(frames, config)
      episode += 1
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
        observation, reward, done = take_action(atari, observation, action,
                                                exploration_bonus,
                                                replay_memory, config)
        episode_score += reward

        # Train on random batch
        batch = replay_memory.sample_batch(config.batch_size, step)
        feed_dict = build_feed_dict(batch, policy_network, target_network,
                                    constraint_network, error_weights)
        if step % config.summary_step_frequency == 0:
          # Don't write summaries every step
          errors, _, summary = session.run([td_errors, train_op, summary_op],
                                           feed_dict)
          summary_writer.add_summary(summary, step)
        else:
          errors, _ = session.run([td_errors, train_op], feed_dict)
        replay_memory.update_priorities(batch.indices, errors)

        # Reset target_action_value network
        if step % config.target_network_update_frequency == 0:
          reset_target_network(action_value_scope, target_action_value_scope)

      # Log episode
      log_episode(episode, start_time, episode_score, episode_steps)


def optimize(loss, global_step, config):
  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = util.add_loss_summaries(loss)

  # Optimizer
  opt = tf.train.AdamOptimizer()

  # Minimize loss
  with tf.control_dependencies([loss_averages_op]):
    grads = opt.compute_gradients(loss)
    if config.grad_clipping:
      grads = [(tf.clip_by_value(grad, -config.grad_clipping,
                                 config.grad_clipping), var)
               for grad, var in grads if grad is not None]
    train_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram('trainable', var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram('gradient', grad)

  return train_op


def process_reward(reward, frames, exploration_bonus, config):
  if exploration_bonus:
    reward += exploration_bonus.bonus(frames)

  if config.reward_clipping:
    reward = max(-config.reward_clipping, min(reward, config.reward_clipping))

  return reward


def process_frames(frames, config):
  observation = []
  for i in range(-config.input_frames, 0):
    image = util.process_image(frames[i - 1:i], config.input_shape)
    image /= 255.0  # Normalize each pixel between 0 and 1
    observation.append(image)
  return observation


def prepopulate_replay_memory(replay_memory, atari, exploration_bonus, config):
  """Play game with random actions to populate the replay memory"""

  count = 0
  done = True

  while count <= config.replay_start_size or not done:
    if done:
      frames, _, done = atari.reset()
      observation = process_frames(frames, config)

    action = atari.sample_action()
    _, _, done = take_action(atari, observation, action, exploration_bonus,
                             replay_memory, config)
    count += 1


def take_action(atari, observation, action, exploration_bonus, replay_memory,
                config):
  frames, reward, done = atari.step(action)

  training_reward = process_reward(reward, frames, exploration_bonus, config)
  replay_memory.store(observation, action, training_reward, done)

  next_observation = process_frames(frames, config)
  return next_observation, reward, done


def epsilon(step, config):
  """Epsilon is linearly annealed from an initial exploration value
  to a final exploration value over a number of steps"""

  annealing_rate = (config.initial_exploration - config.final_exploration
                    ) / config.final_exploration_frame
  annealed_exploration = config.initial_exploration - (step * annealing_rate)
  return max(annealed_exploration, config.final_exploration)


def build_feed_dict(batch, policy_network, target_network, constraint_network,
                    error_weights):
  feed_dict = {
      policy_network.input_frames: batch.observations,
      policy_network.action_input: batch.actions,
      target_network.reward_input: batch.rewards,
      target_network.alive_input: batch.alives,
      target_network.input_frames: batch.next_observations,
      error_weights: batch.error_weights
  }

  if constraint_network:
    constraint_feed_dict = {
        constraint_network.past_input_frames: batch.past_observations,
        constraint_network.past_actions: batch.past_actions,
        constraint_network.past_rewards: batch.past_rewards,
        constraint_network.past_alives: batch.past_alives,
        constraint_network.rewards: batch.rewards,
        constraint_network.alives: batch.alives,
        constraint_network.future_input_frames: batch.future_observations,
        constraint_network.future_rewards: batch.future_rewards,
        constraint_network.future_alives: batch.future_alives,
        constraint_network.total_rewards: batch.total_rewards
    }
    feed_dict.update(constraint_feed_dict)

  return feed_dict


def log_episode(episode, start_time, score, steps):
  now = datetime.strftime(datetime.now(), '%F %X')
  duration = time.time() - start_time
  steps_per_sec = steps / duration
  format_string = ('%s: Episode %d, score %.0f '
                   '(%d steps, %.2f secs, %.2f steps/sec)')
  print(format_string % (now, episode, score, steps, duration, steps_per_sec))


def main(_):
  config = flags.FLAGS
  config.input_shape = eval(config.input_shape)
  config.exploration_image_shape = eval(config.exploration_image_shape)

  train(config)


if __name__ == '__main__':
  tf.app.run()
