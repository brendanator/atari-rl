from agents import Trainer
from atari import Atari
import re
import signal
from sys import stdin
import tensorflow as tf
import util

flags = tf.app.flags

# Environment
flags.DEFINE_string('game', 'SpaceInvaders',
                    'The Arcade Learning Environment game to play')
flags.DEFINE_string('frameskip', 4, 'Number of frames to repeat actions for. '
                    'Can be int or tuple with min and max+1')
flags.DEFINE_float(
    'repeat_action_probability', 0.25,
    'Probability of ignoring the agent action and repeat last action')
flags.DEFINE_string('input_shape', (84, 84), 'Rescale input to this shape')
flags.DEFINE_integer('input_frames', 4, 'Number of frames to input')
flags.DEFINE_integer(
    'max_noops', 30,
    'Maximum number of noop actions to perform at start of episode')

# Agent
flags.DEFINE_bool('double_q', False, 'Enable Double Q-Learning')
flags.DEFINE_bool('sarsa', False, 'Enable SARSA')
flags.DEFINE_bool('bootstrapped', False, 'Whether to use bootstrapped DQN')
flags.DEFINE_integer('num_bootstrap_heads', 10,
                     'Number of bootstrapped head to use')
flags.DEFINE_float('bootstrap_mask_probability', 1.0,
                   'Probability each head has of training on each experience')
flags.DEFINE_bool(
    'bootstrap_use_ensemble', False,
    'Choose action with most bootstrap heads votes. Use only in evaluation')
flags.DEFINE_integer('replay_capacity', 1000000, 'Size of replay memory')
flags.DEFINE_integer(
    'replay_start_size', 50000,
    'Pre-populate the replay memory with this number of random actions')
flags.DEFINE_string(
    'replay_priorities', 'uniform',
    'Prioritized replay memory implementation: [uniform|proportional]')
flags.DEFINE_float('replay_alpha', 0.6,
                   'Prioritized experience replay exponent')
flags.DEFINE_float('replay_beta', 0.4, 'Initial importance sampling exponent')
flags.DEFINE_bool('persistent_advantage_learning', False,
                  'Enable persistent advantage learning')
flags.DEFINE_float('pal_alpha', 0.9, 'Persistent advantage learning alpha')
flags.DEFINE_bool('reward_scaling', False, 'Learn reward scaling')
flags.DEFINE_float('reward_scaling_beta', 1e-4,
                   'Reward scaling exponential moving average coefficient')
flags.DEFINE_float('reward_scaling_stddev', 1,
                   'Reward scaling standard deviation')
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
flags.DEFINE_string('exploration_image_shape', (42, 42),
                    'Shape of image to use with CTS in exploration bonus')

# Training
flags.DEFINE_integer('batch_size', 32, 'Batch size. Ignored for async')
flags.DEFINE_integer('num_steps', 50000000, 'Number of steps to train on')
flags.DEFINE_integer(
    'target_network_update_period', 10000,
    'The number of parameter updates before the target network is updated')
flags.DEFINE_string('async', None, 'Async algorithm [one_step|n_step|a3c]')
flags.DEFINE_float('entropy_beta', 0.01, 'Entropy regularization weight')
flags.DEFINE_integer(
    'num_threads', 16,
    'Number of asynchronous actor learners to run concurrently')
flags.DEFINE_integer('train_period', 4,
                     'The number of steps between training updates')
flags.DEFINE_float('discount_rate', 0.99, 'Discount rate for future rewards')

# Exploration
flags.DEFINE_float('initial_exploration', 1.0,
                   'Initial value of epsilon is epsilon-greedy exploration')
flags.DEFINE_float('final_exploration', 0.1,
                   'Final value of epsilon is epsilon-greedy exploration')
flags.DEFINE_integer(
    'final_exploration_frame', 10000000,
    'The number of frames over to anneal epsilon to its final value')

# Clipping
flags.DEFINE_float('reward_clipping', 1.0,
                   'Range around zero to limit rewards to. 0 to disable. '
                   'Disabled if reward_scaling is True')
flags.DEFINE_float('loss_clipping', 1.0,
                   'Range around zero to limit loss to. 0 to disable')
flags.DEFINE_float('grad_clipping', 10.0,
                   'Range around zero to limit gradients to. 0 to disable')

# Checkpoints/summaries
flags.DEFINE_string(
    'run_dir', None, 'Directory to read/write checkpoints/summaries to. '
    'Defaults to runs/<game>/run_<counter>. Use "latest" to continue previous run'
)
flags.DEFINE_integer('summary_step_period', 100,
                     'How many training steps between writing summaries')

# Render
flags.DEFINE_bool('render', False, 'Show game during training')

# Profiling
flags.DEFINE_bool('profile', False, 'Enable profiling')


def main(_):
  config = create_config()
  trainer = Trainer(config)

  # Register signal handler
  def stop_training(signum, frame):
    print('Do you want to save replay memory? [y/N]')
    save_replay_memory = stdin.readline() == 'y'
    trainer.stop_training(save_replay_memory)

  signal.signal(signal.SIGINT, stop_training)

  if config.profile:
    import cProfile as profile
    profile.runctx('trainer.train()', globals(), locals(), 'main.prof')
  else:
    trainer.train()


def create_config():
  config = flags.FLAGS
  config.game = '_'.join(
      [g.lower() for g in re.findall('[A-Z]?[a-z]+', config.game)])
  config.num_actions = Atari.num_actions(config)
  config.frameskip = eval(str(config.frameskip))
  config.input_shape = eval(str(config.input_shape))
  config.exploration_image_shape = eval(str(config.exploration_image_shape))
  config.reward_clipping = config.reward_clipping and not config.reward_scaling
  config.run_dir = util.run_directory(config)

  if not config.bootstrapped: config.num_bootstrap_heads = 1

  if config.async is None:
    config.num_threads = 1
  else:
    config.replay_capacity = (config.train_period + 2)
    config.replay_start_size = 0

    if config.async == 'one_step':
      config.batch_size = config.train_period
    elif config.async == 'n_step':
      config.batch_size = 1
    elif config.async == 'a3c':
      config.batch_size = 1
    else:
      raise Exception('Unknown asynchronous algorithm: ' + config.async)
  config.n_step = config.async == 'n_step'
  config.actor_critic = config.async == 'a3c'

  return config


if __name__ == '__main__':
  tf.app.run()
