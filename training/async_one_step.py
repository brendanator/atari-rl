import threading
import os
import time
import tensorflow as tf

from agents.agent import Agent
import networks.dqn as dqn
from agents.replay_memory import *

import util


class AsyncOneStepTrainer(object):
  def __init__(self, config):
    self.config = config

    # Reward Scaling
    if config.reward_scaling:
      self.reward_scaling = RewardScaling(config)
    else:
      self.reward_scaling = None

    # Create action-value network
    self.policy_network = dqn.PolicyNetwork(self.reward_scaling, config)

    # Create target action-value network
    if config.actor_critic:
      self.target_network = dqn.PolicyNetwork(
          self.reward_scaling, config, reuse=True)
    else:
      self.target_network = dqn.TargetNetwork(self.policy_network,
                                              self.reward_scaling, config)

    # Create operation to copy variables from policy network to target network
    self.reset_target_network = self.policy_network.copy_to_network(
        self.target_network)

    # Build loss
    self.build_loss()

    # Create training operation
    self.build_train_op()

    # Create agents
    self.agents = []
    for i in range(config.num_threads):
      replay_memory = LinearReplayMemory(config)
      agent = Agent(self.policy_network, replay_memory, config)
      self.agents.append(agent)

    # Summary/checkpoint
    self.checkpoint_dir = os.path.join(config.train_dir, config.game)
    self.summary_writer = tf.summary.FileWriter(self.checkpoint_dir)
    self.summary_op = tf.summary.merge_all()

  # Splitting out loss
  #  - TargetNetwork should not have reward_input,... and double_q stuff
  #  - There should be a functions/classes for:
  #      - action value loss
  #      - value loss
  #      - policy loss
  #      - persistent advantage loss
  #      - optimality tightening should take a loss function/class to apply into the past/future
  #  - Can optimality tightening not be square loss so it can be
  #        incorporated into td_errors for prioritized replay?
  #  - Can inputs 'know' which data they need from a batch??

  # def q_loss(self, policy_network, target_network):
  #   self.reward_input = tf.placeholder(tf.float32, [None], name='reward')
  #   self.alive_input = tf.placeholder(tf.float32, [None], name='alive')
  #   self.steps_offset_input = tf.placeholder(
  #       tf.float32, [None], name='steps_offset')
  #   reward = tf.expand_dims(self.reward_input, axis=1)
  #   alive = tf.expand_dims(self.alive_input, axis=1)
  #   steps_offset = tf.expand_dims(self.steps_offset_input, axis=1)

  #   discount_rate = tf.pow(config.discount_rate, steps_offset)
  #   target_action_values = reward + (alive * discount_rate *
  #                                    target_network.values)
  #   td_error = policy_network.taken_action_values - target_action_values

  #   return td_error

  # def double_q_loss(self, policy_network, target_network):
  #   with tf.variable_scope('double_q'):
  #     # Policy network shouldn't be updated when calculating target values
  #     target_network.max_actions = tf.stop_gradient(
  #         policy_network.max_actions, name='max_actions')

  #     target_network.values = tf.reduce_sum(
  #         tf.one_hot(self.max_actions, config.num_actions) *
  #         self.action_values,
  #         axis=2,
  #         name='values')

  def build_loss(self):
    if self.config.actor_critic:
      td_errors = self.policy_network.values - self.target_network.values
    else:
      # TD-errors are calculated per bootstrap head
      td_errors = (self.target_network.target_action_values +
                   self.policy_network.taken_action_values)

    if self.config.persistent_advantage_learning:
      self.advantage_learning_net = dqn.TargetNetwork(
          self.policy_network, self.reward_scaling, self.config, reuse=True)
      self.next_state_advantage_learning_net = dqn.TargetNetwork(
          self.policy_network, self.reward_scaling, self.config, reuse=True)

      advantage_learning = td_errors - self.config.pal_alpha * (
          self.advantage_learning_net.values -
          self.advantage_learning_net.taken_action_values)

      next_state_advantage_learning = td_errors - self.config.pal_alpha * (
          self.next_state_advantage_learning_net.values -
          self.next_state_advantage_learning_net.taken_action_values)

      persistent_advantage_learning = tf.maximum(
          advantage_learning,
          next_state_advantage_learning,
          name='persistent_advantage_learning')

      td_errors = persistent_advantage_learning

    # Square errors are also calculated per bootstrap head
    square_errors = tf.square(td_errors)

    if self.config.optimality_tightening:
      self.constraint_network = ConstraintNetwork(
          self.policy_network, self.reward_scaling, self.config)
      penalty = self.constraint_network.violation_penalty
      error_rescaling = self.constraint_network.error_rescaling
      square_errors = (square_errors + penalty) / error_rescaling

    # Apply bootstrap mask
    if self.config.bootstrapped and self.config.bootstrap_mask_probability < 1.0:
      self.bootstrap_mask = tf.placeholder(
          tf.float32, [None, self.policy_network.num_heads],
          name='bootstrap_mask')
      td_errors *= self.bootstrap_mask
      square_errors *= self.bootstrap_mask

    # Sum bootstrap heads
    self.td_errors = tf.reduce_sum(td_errors, axis=1, name='td_errors')
    square_error = tf.reduce_sum(square_errors, axis=1, name='square_error')

    # Apply importance sampling
    if self.config.replay_prioritized:
      self.error_weights = tf.placeholder(tf.float32, [None], 'error_weights')
      square_error = self.error_weights * square_error
    self.loss = tf.reduce_mean(square_error, name='loss')

    # Clip loss
    if self.config.loss_clipping > 0:
      self.loss = tf.maximum(
          -self.config.loss_clipping,
          tf.minimum(self.loss, self.config.loss_clipping),
          name='loss')

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
      threads = []
      for i, agent in enumerate(self.agents):
        thread = threading.Thread(
            target=self.async_one_step_q, args=(session, agent))
        thread.name = 'Agent-%d' % (i + 1)
        thread.start()
        threads.append(thread)

      for thread in threads:
        thread.join()

  def async_one_step_q(self, session, agent):
    # Initialize step count
    step = 0

    while step < self.config.num_steps:
      # Start new episode
      observation, _, done = agent.new_game()

      # Play until losing
      while not done:
        # Reset target action-value network
        if self.should_reset(step):
          session.run(self.reset_target_network)

        # Choose next action
        action = agent.action(observation, step, session)

        # Take action
        observation, _, done = agent.take_action(action)

        if agent.replay_memory.available() == self.config.batch_size:
          # Train on random batch
          step = self.train_step(session, agent.replay_memory, step)

          # Clear replay memory
          agent.replay_memory.clear()

      # Log episode
      agent.log_episode()

  def should_reset(self, step):
    if self.config.async == 'a3c':
      return False
    else:
      return step % self.config.target_network_update_period == 0

  def train_step(self, session, replay_memory, step):
    batch = replay_memory.sample_batch(self.config.batch_size, step)
    feed_dict = self.build_feed_dict(batch)

    if step % self.config.summary_step_period == 0:
      global_step, td_errors, _, summary = session.run(
          [self.global_step, self.td_errors, self.train_op, self.summary_op],
          feed_dict)
      self.summary_writer.add_summary(summary, step)
    else:
      global_step, td_errors, _ = session.run(
          [self.global_step, self.td_errors, self.train_op], feed_dict)

    replay_memory.update_priorities(batch.indices, td_errors)

    return global_step

  def build_feed_dict(self, batch):
    feed_dict = {
        self.policy_network.input_frames: batch.observations(),
        self.policy_network.action_input: batch.actions(),
        self.target_network.reward_input: batch.rewards(),
        self.target_network.alive_input: batch.alives(),
        self.target_network.input_frames: batch.observations(offset=1),
        self.target_network.action_input: batch.actions(offset=1),
        self.target_network.steps_offset_input: tf.ones_like(batch.indices),
    }

    if self.config.async == 'n_step' or self.config.async == 'a3c':
      rewards = batch.rewards()
      reward = 0
      for i in np.flip(batch.indices, axis=0):
        reward = reward * self.config.discount_rate + rewards[i]
        rewards[i] = reward

      n = self.config.batch_size
      alives = np.tile(batch.alives(offset=n)[0], reps=n)
      input_frames = np.tile(
          batch.observations(offset=n)[0], reps=(n, 1, 1, 1))
      actions = np.tile(batch.actions(offset=n)[0], reps=n)

      n_step_feed_dict = {
          self.target_network.reward_input: rewards,
          self.target_network.alive_input: alives,
          self.target_network.input_frames: input_frames,
          self.target_network.action_input: actions,
          self.target_network.steps_offset_input: batch.indices + 1,
      }
      feed_dict.update(n_step_feed_dict)

    if self.config.optimality_tightening:
      steps = config.optimality_tightening_steps + 1
      constraint_feed_dict = {
          self.constraint_network.past_input_frames:
          batch.observations(-1, -steps),
          self.constraint_network.past_actions: batch.actions(-1, -steps),
          self.constraint_network.past_rewards: batch.rewards(-1, -steps),
          self.constraint_network.past_alives: batch.alives(-1, -steps),
          self.constraint_network.rewards: batch.rewards(),
          self.constraint_network.alives: batch.alives(),
          self.constraint_network.future_input_frames:
          batch.observations(1, steps),
          self.constraint_network.future_rewards: batch.rewards(1, steps),
          self.constraint_network.future_alives: batch.alives(1, steps),
          self.constraint_network.total_rewards: batch.total_rewards()
      }
      feed_dict.update(constraint_feed_dict)

    if self.config.replay_prioritized:
      feed_dict[self.error_weights] = batch.error_weights()

    if self.config.bootstrapped and self.config.bootstrap_mask_probability < 1.0:
      feed_dict[self.bootstrap_mask] = batch.bootstrap_mask()

    if self.config.persistent_advantage_learning:
      persistent_advantage_feed_dict = {
          self.advantage_learning_net.input_frames: batch.observations(),
          self.advantage_learning_net.action_input: batch.actions(),
          self.next_state_advantage_learning_net.input_frames:
          batch.observations(offset=1),
          self.next_state_advantage_learning_net.action_input: batch.actions()
      }
      feed_dict.update(persistent_advantage_feed_dict)

    if self.reward_scaling:
      feed_dict[self.reward_scaling.sigma_squared_input] = (
          self.reward_scaling.sigma_squared(batch.rewards()))

    return feed_dict
