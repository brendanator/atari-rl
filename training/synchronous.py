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
    # self.policy_network = dqn.PolicyNetwork(self.reward_scaling, config)
    self.policy_network = self.factory.policy_network()

    # Create target action-value network
    # self.target_network = dqn.TargetNetwork(self.policy_network,
    #                                         self.reward_scaling, config)
    self.target_network = self.factory.target_network(t=1)

    # Create operation to copy variables from policy network to target network
    self.reset_target_network = self.policy_network.copy_to_network(
        self.target_network)

    # Build loss
    self.build_loss()

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

  def build_loss(self):
    self.losses = Losses(self.factory, self.config)
    self.td_errors = tf.identity(self.losses.loss(), name='td_errors')
    self.loss = tf.reduce_mean(tf.square(self.td_errors), name='loss')

  # def build_loss(self):
  #   # TD-errors are calculated per bootstrap head
  #   td_errors = (self.target_network.target_action_values +
  #                self.policy_network.taken_action_values)

  #   if self.config.persistent_advantage_learning:
  #     self.advantage_learning_net = dqn.TargetNetwork(
  #         self.policy_network, self.reward_scaling, self.config, reuse=True)
  #     self.next_state_advantage_learning_net = dqn.TargetNetwork(
  #         self.policy_network, self.reward_scaling, self.config, reuse=True)

  #     advantage_learning = td_errors - self.config.pal_alpha * (
  #         self.advantage_learning_net.values -
  #         self.advantage_learning_net.taken_action_values)

  #     next_state_advantage_learning = td_errors - self.config.pal_alpha * (
  #         self.next_state_advantage_learning_net.values -
  #         self.next_state_advantage_learning_net.taken_action_values)

  #     persistent_advantage_learning = tf.maximum(
  #         advantage_learning,
  #         next_state_advantage_learning,
  #         name='persistent_advantage_learning')

  #     td_errors = persistent_advantage_learning

  #   # Square errors are also calculated per bootstrap head
  #   square_errors = tf.square(td_errors)

  #   if self.config.optimality_tightening:
  #     self.constraint_network = ConstraintNetwork(
  #         self.policy_network, self.reward_scaling, self.config)
  #     penalty = self.constraint_network.violation_penalty
  #     error_rescaling = self.constraint_network.error_rescaling
  #     square_errors = (square_errors + penalty) / error_rescaling

  #   # Apply bootstrap mask
  #   if self.config.bootstrapped and self.config.bootstrap_mask_probability < 1.0:
  #     self.bootstrap_mask = tf.placeholder(
  #         tf.float32, [None, self.policy_network.num_heads],
  #         name='bootstrap_mask')
  #     td_errors *= self.bootstrap_mask
  #     square_errors *= self.bootstrap_mask

  #   # Sum bootstrap heads
  #   self.td_errors = tf.reduce_sum(td_errors, axis=1, name='td_errors')
  #   square_error = tf.reduce_sum(square_errors, axis=1, name='square_error')

  #   # Apply importance sampling
  #   self.error_weights = tf.placeholder(tf.float32, [None], 'error_weights')
  #   self.loss = tf.reduce_mean(self.error_weights * square_error, name='loss')

  #   # Clip loss
  #   if self.config.loss_clipping > 0:
  #     self.loss = tf.maximum(
  #         -self.config.loss_clipping,
  #         tf.minimum(self.loss, self.config.loss_clipping),
  #         name='loss')

  def build_train_op(self):
    # Create global step
    self.global_step = tf.contrib.framework.get_or_create_global_step()

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = util.add_loss_summaries(self.loss)

    # Optimizer
    opt = tf.train.AdamOptimizer()

    # Minimize loss
    with tf.control_dependencies([loss_averages_op]):
      # Variables to update
      policy_variables = self.policy_network.variables
      if self.reward_scaling:
        reward_scaling_variables = self.reward_scaling.variables
      else:
        reward_scaling_variables = []

      # Compute gradients
      grads = opt.compute_gradients(
          self.loss, var_list=policy_variables + reward_scaling_variables)

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
      self.train_op = opt.apply_gradients(grads, global_step=self.global_step)

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
      fetches = [
          self.global_step, self.td_errors, self.train_op, self.summary_op
      ]
      global_step, td_errors, _, summary = session.run(
          fetches, batch.build_feed_dict(fetches))
      self.summary_writer.add_summary(summary, step)
    else:
      fetches = [self.global_step, self.td_errors, self.train_op]
      global_step, td_errors, _ = session.run(fetches,
                                              batch.build_feed_dict(fetches))

    self.replay_memory.update_priorities(batch.indices, td_errors)

    return global_step

  def build_feed_dict(self, batch):
    feed_dict = {
        self.policy_network.input_frames: batch.observations,
        self.policy_network.action_input: batch.actions,
        self.target_network.reward_input: batch.rewards,
        self.target_network.alive_input: batch.alives,
        self.target_network.input_frames: batch.next_observations,
        self.target_network.action_input: batch.next_actions,
        self.error_weights: batch.error_weights
    }

    if self.config.optimality_tightening:
      constraint_feed_dict = {
          self.constraint_network.past_input_frames: batch.past_observations,
          self.constraint_network.past_actions: batch.past_actions,
          self.constraint_network.past_rewards: batch.past_rewards,
          self.constraint_network.past_alives: batch.past_alives,
          self.constraint_network.rewards: batch.rewards,
          self.constraint_network.alives: batch.alives,
          self.constraint_network.future_input_frames:
          batch.future_observations,
          self.constraint_network.future_rewards: batch.future_rewards,
          self.constraint_network.future_alives: batch.future_alives,
          self.constraint_network.total_rewards: batch.total_rewards
      }
      feed_dict.update(constraint_feed_dict)

    if self.config.bootstrapped and self.config.bootstrap_mask_probability < 1.0:
      feed_dict[self.bootstrap_mask] = batch.bootstrap_mask

    if self.config.persistent_advantage_learning:
      persistent_advantage_feed_dict = {
          self.advantage_learning_net.input_frames: batch.observations,
          self.advantage_learning_net.action_input: batch.actions,
          self.next_state_advantage_learning_net.input_frames:
          batch.next_observations,
          self.next_state_advantage_learning_net.action_input: batch.actions
      }
      feed_dict.update(persistent_advantage_feed_dict)

    if self.reward_scaling:
      feed_dict[self.reward_scaling.sigma_squared_input] = (
          self.reward_scaling.sigma_squared(batch.rewards))

    return feed_dict
