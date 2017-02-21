import numpy as np
import os
import tensorflow as tf

from .agent import Agent
from .replay_memory import ReplayMemory
from networks import Losses, NetworkFactory
import util


class Trainer(object):
  def __init__(self, config):
    self.config = config

    # Create network factory
    self.factory = NetworkFactory(config)
    self.reward_scaling = self.factory.reward_scaling

    # Create action-value network
    self.policy_network = self.factory.policy_network()

    # Build loss
    self.losses = Losses(self.factory, config)

    # Create training operation
    self.build_train_op()

    # Create agent
    self.replay_memory = ReplayMemory(config)
    self.agent = Agent(self.policy_network, self.replay_memory, config)
    self.agent.populate_replay_memory()

    # Build reset operation
    self.factory.create_reset_target_network_op()

    # Summary/checkpoint
    self.checkpoint_dir = os.path.join(config.train_dir, config.game)
    self.summary_writer = tf.summary.FileWriter(self.checkpoint_dir)
    self.summary_op = tf.summary.merge_all()

  def build_train_op(self):
    # Optimizer
    opt = tf.train.AdamOptimizer()

    # Compute gradients
    policy_vars = self.policy_network.variables
    reward_scaling_vars = self.reward_scaling.variables
    grads = opt.compute_gradients(
        self.losses.loss, var_list=policy_vars + reward_scaling_vars)

    # Apply normalized SGD for reward scaling
    grads = self.reward_scaling.scale_gradients(grads, policy_vars)

    # Clip gradients
    if self.config.grad_clipping:
      grads = [(tf.clip_by_value(grad, -self.config.grad_clipping,
                                 self.config.grad_clipping), var)
               for grad, var in grads if grad is not None]

    # Create training op
    loss_summaries = util.add_loss_summaries(self.losses.loss)
    global_step = tf.contrib.framework.get_or_create_global_step()
    minimize = opt.apply_gradients(grads, global_step=global_step)
    with tf.control_dependencies([loss_summaries, minimize]):
      self.train_op = tf.identity(self.losses.td_error, name='train')

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
      tf.summary.histogram('trainable', var)

    # Add histograms for gradients.
    for grad, var in grads:
      if grad is not None:
        tf.summary.histogram('gradient', grad)

  def train(self):
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=self.checkpoint_dir,
        save_summaries_steps=0  # Summaries will be saved with train_op only
    ) as session:

      # Initialize step count
      step = 0

      while step < self.config.num_steps:
        # Start new episode
        observation, _, done = self.agent.new_game()

        # Play until losing
        while not done:
          self.factory.reset_target_network(session, step)
          action = self.agent.action(session, step, observation)
          observation, _, done = self.agent.take_action(action)
          self.train_batch(session, step)
          step += 1

        # Log episode
        self.agent.log_episode()

  def train_batch(self, session, step):
    if step % self.config.train_period > 0: return

    batch = self.replay_memory.sample_batch(self.config.batch_size, step)

    if step % self.config.summary_step_period == 0:
      fetches = [self.train_op, self.summary_op]
      feed_dict = batch.build_feed_dict(fetches)
      td_errors, summary = session.run(fetches, feed_dict)
      self.summary_writer.add_summary(summary, step)
    else:
      fetches = self.train_op
      feed_dict = batch.build_feed_dict(fetches)
      td_errors = session.run(fetches, feed_dict)

    self.replay_memory.update_priorities(batch.indices, td_errors)
