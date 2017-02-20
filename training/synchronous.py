import os

import tensorflow as tf
import numpy as np

from agents.agent import Agent
from networks import Losses, NetworkFactory
import util
from agents.optimality_tightening import ConstraintNetwork
from agents.replay_memory import ReplayMemory
from agents.reward_scaling import RewardScaling


class SynchronousTrainer(object):
  def __init__(self, config):
    self.config = config

    # Reward Scaling TODO Move into factory?
    if config.reward_scaling:
      self.reward_scaling = RewardScaling(config)
    else:
      self.reward_scaling = None

    # Create network factory
    self.factory = NetworkFactory(self.reward_scaling, config)

    # Create action-value network
    self.policy_network = self.factory.policy_network()

    # Create target action-value network
    self.target_network = self.factory.target_network(t=1)

    # Create operation to copy variables from policy network to target network
    self.reset_target_network = self.policy_network.copy_to_network(
        self.target_network)

    # Build loss
    self.losses = Losses(self.factory, config)

    # Create training operation
    self.build_train_op()

    # Create agent
    self.replay_memory = ReplayMemory(config)
    self.agent = Agent(self.policy_network, self.replay_memory, config)
    self.agent.populate_replay_memory()

    # Summary/checkpoint
    self.checkpoint_dir = os.path.join(config.train_dir, config.game)
    self.summary_writer = tf.summary.FileWriter(self.checkpoint_dir)
    self.summary_op = tf.summary.merge_all()

  def build_train_op(self):
    # Create global step
    self.global_step = tf.contrib.framework.get_or_create_global_step()

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = util.add_loss_summaries(self.losses.loss)

    # Optimizer
    opt = tf.train.AdamOptimizer()

    # Minimize loss
    with tf.control_dependencies([loss_averages_op]):
      # Variables to update
      policy_variables = self.policy_network.variables
      if self.reward_scaling:  # TODO Do something with reward_scaling
        reward_scaling_variables = self.reward_scaling.variables
      else:
        reward_scaling_variables = []

      # Compute gradients
      grads = opt.compute_gradients(
          self.losses.loss,
          var_list=policy_variables + reward_scaling_variables)

      # Apply normalized SGD for reward scaling
      if self.reward_scaling:
        grads_ = []
        for grad, var in grads:
          if grad is not None:
            if var in policy_variables:
              grad /= self.reward_scaling.sigma_squared_input
            grads_.append((grad, var))
        grads = grads_

      # Clip gradients
      if self.config.grad_clipping:
        grads = [(tf.clip_by_value(grad, -self.config.grad_clipping,
                                   self.config.grad_clipping), var)
                 for grad, var in grads if grad is not None]

      # Create training op
      minimize = opt.apply_gradients(grads, global_step=self.global_step)
      with tf.control_dependencies([minimize]):
        self.train_op = self.losses.td_errors

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
          # Reset target action-value network
          if step % self.config.target_network_update_period == 0:
            session.run(self.reset_target_network)

          # Choose next action
          action = self.agent.action(observation, step, session)

          # Take action
          observation, _, done = self.agent.take_action(action)

          # Train on random batch
          if step % self.config.train_period == 0:
            step = self.train_step(session, step)

          # Increment step
          step += 1

        # Log episode
        self.agent.log_episode()

  def train_step(self, session, step):
    batch = self.replay_memory.sample_batch(self.config.batch_size, step)

    if step % self.config.summary_step_period == 0:
      fetches = [self.global_step, self.train_op, self.summary_op]
      feed_dict = batch.build_feed_dict(fetches)
      global_step, td_errors, summary = session.run(fetches, feed_dict)
      self.summary_writer.add_summary(summary, step)
    else:
      fetches = [self.global_step, self.train_op]
      feed_dict = batch.build_feed_dict(fetches)
      global_step, td_errors = session.run(fetches, feed_dict)

    self.replay_memory.update_priorities(batch.indices, td_errors)

    return global_step
