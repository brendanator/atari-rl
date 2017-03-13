import tensorflow as tf
import util


class Losses(object):
  def __init__(self, factory, config):
    self.config = config
    self.setup_dsl(factory, config)
    self.build_loss(config)

  def build_loss(self, config):
    # Basic loss
    if self.config.n_step:
      priorities, loss = self.n_step_loss()
    elif self.config.actor_critic:
      priorities, loss = self.actor_critic_loss()
    else:
      priorities, loss = self.one_step_loss()

    # Priorities
    self.priorities = tf.identity(priorities, name='priorities')

    # Optimality Tightening
    if config.optimality_tightening:
      penalty, error_rescaling = self.optimality_tightening()
      loss = (loss + penalty) / error_rescaling

    # Apply bootstrap mask
    if config.bootstrapped and config.bootstrap_mask_probability < 1.0:
      loss *= self.bootstrap_mask

    # Sum bootstrap heads
    loss = tf.reduce_sum(loss, axis=1, name='square_error')

    # Apply priority weights
    if config.replay_priorities != 'uniform':
      importance_sampling = 1 / (self.replay_count *
                                 self.priority_probabilities)

      beta_grad = (1.0 - config.replay_beta) / config.num_steps
      beta = config.replay_beta + beta_grad * self.global_step
      priority_weights = importance_sampling**beta

      loss *= priority_weights / tf.reduce_max(priority_weights)

    # Clip loss
    self.loss = tf.reduce_mean(loss, name='loss')
    if config.loss_clipping > 0:
      self.loss = tf.clip_by_value(
          self.loss,
          -config.loss_clipping,
          config.loss_clipping,
          name='clipped_loss')

  def one_step_loss(self):
    with tf.name_scope('one_step_loss'):
      taken_action_value = self.policy_network[0].taken_action_value

      if self.config.persistent_advantage_learning:
        target_value = self.persistent_advantage_target()
      else:
        target_value = self.one_step_target()

      error = target_value - taken_action_value
      loss = tf.square(error)

      tf.summary.scalar('Q_t', tf.reduce_mean(taken_action_value))
      tf.summary.scalar('y_t', tf.reduce_mean(target_value))

      return error, loss

  def one_step_target(self):
    with tf.name_scope('one_step_target'):
      return self.reward[0] + self.discount * self.value(1)

  def value(self, t):
    if self.config.double_q:
      with tf.name_scope('double_q_value'):
        greedy_action = tf.stop_gradient(self.policy_network[t].greedy_action)
        return self.target_network[t].action_value(greedy_action)
    elif self.config.sarsa:
      with tf.name_scope('sarsa_value'):
        return self.target_network[t].taken_action_value
    else:
      with tf.name_scope('q_value'):
        return self.target_network[t].value

  def persistent_advantage_target(self):
    with tf.name_scope('persistent_advantage_target'):
      one_step_target = self.one_step_target()
      alpha = self.config.pal_alpha
      action = self.action[0]

      with tf.name_scope('advantage_target'):
        advantage = self.value(0) - self.target_network[0].action_value(action)
        advantage_target = one_step_target - alpha * advantage

      with tf.name_scope('next_advantage_target'):
        next_advantage = (
            self.value(1) - self.target_network[1].action_value(action))
        next_advantage_target = one_step_target - alpha * next_advantage

      return tf.maximum(advantage_target, next_advantage_target, name='target')

  def optimality_tightening(self):
    with tf.name_scope('optimality_tightening'):
      taken_action_value = self.policy_network[0].taken_action_value

      # Upper bounds
      upper_bounds = []
      rewards = 0
      for t in range(-1, -self.config.optimality_tightening_steps - 1, -1):
        with tf.name_scope(util.format_offset('upper_bound', t)):
          rewards = self.reward[t] + self.discount * rewards
          q_value = (self.discounts[t] *
                     self.target_network[t].taken_action_value)
          upper_bound = q_value - rewards
        upper_bounds.append(upper_bound)

      upper_bound = tf.reduce_min(tf.stack(upper_bounds, axis=2), axis=2)
      upper_bound_difference = taken_action_value - upper_bound
      upper_bound_breached = tf.to_float(upper_bound_difference > 0)
      upper_bound_penalty = tf.square(tf.nn.relu(upper_bound_difference))

      # Lower bounds
      discounted_reward = tf.tile(
          self.discounted_reward[0],
          multiples=[1, self.config.num_bootstrap_heads])
      lower_bounds = [discounted_reward]
      rewards = self.reward[0]
      for t in range(1, self.config.optimality_tightening_steps + 1):
        with tf.name_scope(util.format_offset('lower_bound', t)):
          rewards += self.reward[t] * self.discounts[t]
          lower_bound = rewards + self.discounts[t + 1] * self.value(t + 1)
        lower_bounds.append(lower_bound)

      lower_bound = tf.reduce_max(tf.stack(lower_bounds, axis=2), axis=2)
      lower_bound_difference = lower_bound - taken_action_value
      lower_bound_breached = tf.to_float(lower_bound_difference > 0)
      lower_bound_penalty = tf.square(tf.nn.relu(lower_bound_difference))

      # Penalty and rescaling
      penalty = self.config.optimality_penalty_ratio * (
          lower_bound_penalty + upper_bound_penalty)
      constraints_breached = lower_bound_breached + upper_bound_breached
      error_rescaling = 1.0 / (
          1.0 + constraints_breached * self.config.optimality_penalty_ratio)

      tf.summary.scalar('discounted_reward', tf.reduce_mean(discounted_reward))
      tf.summary.scalar('lower_bound', tf.reduce_mean(lower_bound))
      tf.summary.scalar('upper_bound', tf.reduce_mean(upper_bound))

      return penalty, error_rescaling

  def n_step_loss(self):
    n = self.config.train_period

    loss = 0
    reward = self.policy_network[n].value

    for i in range(n - 1, -1, -1):
      reward = self.reward[i] + self.discount * reward
      value = self.policy_network[i].value
      td_error = reward - value

      loss += tf.square(td_error)

    return loss, loss

  def actor_critic_loss(self):
    n = self.config.train_period
    entropy_beta = self.config.entropy_beta

    policy_loss, value_loss = 0, 0
    reward = self.policy_network[n].value

    for i in range(n - 1, -1, -1):
      policy_network = self.policy_network[i]

      reward = self.reward[i] + self.discount * reward
      value = policy_network.value
      td_error = reward - value
      log_policy = policy_network.log_policy(self.action[i])

      policy_loss += (log_policy * tf.stop_gradient(td_error) + entropy_beta *
                      policy_network.entropy)
      value_loss += tf.square(td_error)

    loss = policy_loss + value_loss
    return loss, loss

  def setup_dsl(self, factory, config):
    class ArraySyntax(object):
      def __init__(self, getitem):
        self.getitem = getitem

      def __getitem__(self, key):
        return self.getitem(key)

    inputs = factory.inputs

    self.discount = config.discount_rate
    self.discounts = ArraySyntax(lambda t: self.discount**t)
    self.global_step = tf.to_float(inputs.global_step)
    self.replay_count = tf.to_float(inputs.replay_count)
    self.bootstrap_mask = inputs.bootstrap_mask
    self.priority_probabilities = inputs.priority_probabilities

    self.policy_network = ArraySyntax(lambda t: factory.policy_network(t))
    self.target_network = ArraySyntax(lambda t: factory.target_network(t))
    self.action = ArraySyntax(lambda t: inputs.offset_input(t).action)
    self.reward = ArraySyntax(lambda t: inputs.offset_input(t).reward)

    # This is the total discounted reward from the
    # current timestep until the end of the episode
    self.discounted_reward = ArraySyntax(
        lambda t: inputs.offset_input(t).discounted_reward)
