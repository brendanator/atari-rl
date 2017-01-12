import tensorflow as tf
from .dqn import TargetNetwork


class ConstraintNetwork:
  def __init__(self, config):
    self.bound_steps = config.optimality_tightening_steps
    self.penalty_coefficient = config.optimality_penalty_coefficient
    self.lower_bound = self.build_lower_bound(config)
    self.upper_bound = self.build_upper_bound(config)

  def build_upper_bound(self, config):
    self.past_input_frames = tf.placeholder(tf.float32, [
        None, self.bound_steps, config.input_frames, config.input_height,
        config.input_width
    ], 'past_input_frames')
    past_input_frames = tf.unstack(self.past_input_frames, axis=1)

    self.past_actions = tf.placeholder(tf.int32, [None, self.bound_steps])
    past_actions = tf.unstack(self.past_actions, axis=1)

    self.past_rewards = tf.placeholder(tf.float32, [None, self.bound_steps])
    past_rewards = tf.unstack(self.past_rewards, axis=1)

    self.past_discounts = tf.placeholder(tf.float32, [None, self.bound_steps])
    past_discounts = tf.unstack(self.past_discounts, axis=1)

    upper_bounds = []
    for past_input_frame, past_action, past_reward, past_discount in zip(
        past_input_frames, past_actions, past_rewards, past_discounts):

      past_network = TargetNetwork(
          config,
          reuse=True,
          input_frames=past_input_frame,
          action_input=past_action)

      upper_bound = (
          past_discount * past_network.taken_action_value - past_reward)
      upper_bounds.append(upper_bound)

    upper_bounds = tf.pack(upper_bounds, axis=1)
    upper_bound = tf.reduce_min(
        upper_bounds, reduction_indices=1, name='upper_bound')

    return upper_bound

  def build_lower_bound(self, config):
    self.future_input_frames = tf.placeholder(tf.float32, [
        None, self.bound_steps, config.input_frames, config.input_height,
        config.input_width
    ], 'future_input_frames')
    future_input_frames = tf.unstack(self.future_input_frames, axis=1)

    self.future_rewards = tf.placeholder(tf.float32, [None, self.bound_steps])
    future_rewards = tf.unstack(self.future_rewards, axis=1)

    self.future_dones = tf.placeholder(tf.float32, [None, self.bound_steps])
    future_dones = tf.unstack(self.future_dones, axis=1)

    self.future_discounts = tf.placeholder(tf.float32,
                                           [None, self.bound_steps])
    future_discounts = tf.unstack(self.future_discounts, axis=1)

    lower_bounds = []
    for future_input_frame, future_reward, future_done, future_discount in zip(
        future_input_frames, future_rewards, future_dones, future_discounts):

      future_network = TargetNetwork(
          config, reuse=True, input_frames=future_input_frame)

      lower_bound = (
          future_reward +
          (1.0 - future_done) * future_discount * future_network.max_value)

      lower_bounds.append(lower_bound)

    self.total_reward = tf.placeholder(tf.float32, [None], 'total_reward')
    lower_bounds = tf.pack(lower_bounds + [self.total_reward], axis=1)
    lower_bound = tf.reduce_max(
        lower_bounds, reduction_indices=1, name='lower_bound')

    return lower_bound

  def violation_penalty(self, policy_network):
    lower_bound = tf.stop_gradient(self.lower_bound)
    lower_bound_difference = lower_bound - policy_network.taken_action_value
    lower_bound_breached = tf.to_float(lower_bound_difference > 0)
    lower_bound_penalty = tf.square(tf.nn.relu(lower_bound_difference))

    upper_bound = tf.stop_gradient(self.upper_bound)
    upper_bound_difference = policy_network.taken_action_value - upper_bound
    upper_bound_breached = tf.to_float(upper_bound_difference > 0)
    upper_bound_penalty = tf.square(tf.nn.relu(upper_bound_difference))

    violation_penalty = lower_bound_penalty + upper_bound_penalty

    constraint_breaches = lower_bound_breached + upper_bound_breached
    loss_rescaling = 1.0 / (
        1.0 + constraint_breaches * self.penalty_coefficient)

    return violation_penalty, loss_rescaling
