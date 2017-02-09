import tensorflow as tf
import numpy as np
import math

from .dqn import TargetNetwork


class ConstraintNetwork(object):
  def __init__(self, policy_network, config):
    self.policy_network = policy_network
    self.constraint_steps = config.optimality_tightening_steps

    lower_bound = self.build_lower_bound(policy_network, config)
    lower_bound_difference = (
        lower_bound - policy_network.heads_taken_action_value)
    lower_bound_breached = tf.to_float(lower_bound_difference > 0)
    lower_bound_penalty = tf.square(tf.nn.relu(lower_bound_difference))

    upper_bound = self.build_upper_bound(policy_network, config)
    upper_bound_difference = (
        policy_network.heads_taken_action_value - upper_bound)
    upper_bound_breached = tf.to_float(upper_bound_difference > 0)
    upper_bound_penalty = tf.square(tf.nn.relu(upper_bound_difference))

    self.violation_penalty = lower_bound_penalty + upper_bound_penalty

    constraint_breaches = lower_bound_breached + upper_bound_breached
    self.error_rescaling = 1.0 / (
        1.0 + constraint_breaches * config.optimality_penalty_ratio)

  def build_upper_bound(self, policy_network, config):
    # Input frames
    self.past_input_frames = tf.placeholder(
        tf.float32, [None, self.constraint_steps, config.input_frames
                     ] + config.input_shape, 'past_input_frames')
    past_input_frames = tf.unstack(self.past_input_frames, axis=1)

    # Discounts
    past_steps = np.arange(-1, -self.constraint_steps - 1, -1)
    past_discounts = config.discount_rate**past_steps

    # Actions
    self.past_actions = tf.placeholder(tf.int32, [None, self.constraint_steps],
                                       'past_actions')
    past_actions = tf.unstack(self.past_actions, axis=1)

    # Rewards
    self.past_rewards = tf.placeholder(
        tf.float32, [None, self.constraint_steps], 'past_rewards')
    past_rewards = tf.cumsum(self.past_rewards * past_discounts, axis=1)
    past_rewards = tf.unstack(past_rewards, axis=1)

    # Alive
    self.past_alives = tf.placeholder(
        tf.float32, [None, self.constraint_steps], 'past_alives')
    past_alives = tf.expand_dims(
        self.alives, axis=1) * tf.cumprod(
            self.past_alives, axis=1)
    past_alives = tf.equal(past_alives, 1.0)
    past_alives = tf.unstack(past_alives, axis=1)

    # Calculate upper bounds
    upper_bounds = []
    for past_input_frame, past_action, past_reward, past_alive, past_discount in zip(
        past_input_frames, past_actions, past_rewards, past_alives,
        past_discounts):

      past_network = TargetNetwork(
          policy_network,
          config,
          reuse=True,
          input_frames=past_input_frame,
          action_input=past_action)

      past_reward = tf.expand_dims(past_reward, axis=1)
      upper_bound = (
          past_discount * past_network.heads_taken_action_value - past_reward)

      # Ignore upper bound if game hasn't started yet
      upper_bound = tf.select(past_alive, upper_bound,
                              tf.ones_like(upper_bound) * float('inf'))

      upper_bounds.append(upper_bound)

    # Return the minimum upper bound
    upper_bounds = tf.pack(upper_bounds, axis=1)
    return tf.reduce_min(upper_bounds, axis=1, name='upper_bound')

  def build_lower_bound(self, policy_network, config):
    # Input frames
    self.future_input_frames = tf.placeholder(
        tf.float32, [None, self.constraint_steps, config.input_frames
                     ] + config.input_shape, 'future_input_frames')
    future_input_frames = tf.unstack(self.future_input_frames, axis=1)

    # Future discounts for calculating rewards
    future_steps = np.arange(1, self.constraint_steps + 1)
    future_discounts = config.discount_rate**future_steps

    # Rewards
    self.rewards = tf.placeholder(tf.float32, [None], 'rewards')
    self.future_rewards = tf.placeholder(
        tf.float32, [None, self.constraint_steps], 'future_rewards')
    future_rewards = tf.expand_dims(
        self.rewards, axis=1) + tf.cumsum(
            self.future_rewards * future_discounts, axis=1)
    future_rewards = tf.unstack(future_rewards, axis=1)

    # Alive
    self.alives = tf.placeholder(tf.float32, [None], 'alives')
    self.future_alives = tf.placeholder(
        tf.float32, [None, self.constraint_steps], 'future_alives')
    future_alives = tf.expand_dims(
        self.alives, axis=1) * tf.cumprod(
            self.future_alives, axis=1)
    future_alives = tf.equal(future_alives, 1.0)
    future_alives = tf.unstack(future_alives, axis=1)

    # Action values are evaluated a further timestep into the future
    future_discounts = future_discounts * config.discount_rate

    # Calculate lower bounds
    lower_bounds = []
    for future_input_frame, future_reward, future_alive, future_discount in zip(
        future_input_frames, future_rewards, future_alives, future_discounts):

      future_network = TargetNetwork(
          policy_network, config, reuse=True, input_frames=future_input_frame)

      future_reward = tf.expand_dims(future_reward, axis=1)
      lower_bound = (
          future_reward + future_discount * future_network.heads_max_value)

      # Ignore lower bound if game is finished
      lower_bound = tf.select(future_alive, lower_bound,
                              tf.ones_like(lower_bound) * float('-inf'))

      lower_bounds.append(lower_bound)

    # Add total discounted reward as an additional lower bound
    self.total_rewards = tf.placeholder(tf.float32, [None], 'total_reward')
    total_rewards = tf.tile(
        tf.expand_dims(
            self.total_rewards, axis=1),
        multiples=[1, policy_network.num_heads])
    lower_bounds = tf.pack(lower_bounds + [total_rewards], axis=1)

    # Return the maximum lower bound
    return tf.reduce_max(lower_bounds, axis=1, name='lower_bound')
