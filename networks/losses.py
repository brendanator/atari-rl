import tensorflow as tf


class ArraySyntax(object):
  def __init__(self, getitem):
    self.getitem = getitem

  def __getitem__(self, key):
    return self.getitem(key)


class Losses(object):
  def __init__(self, factory, config):
    self.discount = config.discount_rate
    self.config = config

    self.reward = ArraySyntax(lambda t: tf.expand_dims(factory.inputs(t).reward_input, axis=1))
    self.action = ArraySyntax(lambda t: tf.expand_dims(factory.inputs(t).action_input, axis=1))
    self.policy_network = ArraySyntax(lambda t: factory.policy_network(t))
    self.target_network = ArraySyntax(lambda t: factory.target_network(t))

  def loss(self):
    if self.config.double_q:
      return self.double_q_loss()
    else:
      return self.q_loss()

  def q_loss(self, t=0):
    taken_action_value = self.policy_network[t].action_value(self.action[t])
    target = self.reward[t] + self.discount * self.target_network[t + 1].values
    return taken_action_value - target

  def double_q_loss(self, t=0):
    taken_action_value = self.policy_network[t].action_value(self.action[t])

    greedy_action = self.policy_network[t + 1].greedy_actions
    target_value = self.target_network[t + 1].action_value(greedy_action)
    target = self.reward[t] + self.discount * target_value

    return taken_action_value - target

  def sarsa_loss(self, t):
    taken_action_value = self.policy_network[t].action_value(self.action[t])

    next_action = self.action[t + 1]
    next_action_value = self.target_network[t + 1].action_values(next_action)
    target = self.reward[t] + self.discount * next_action_values

    return taken_action_value - target

  def actor_critic_loss(self, t, n):
    policy_loss, value_loss = 0, 0

    reward = self.policy_net.value(t + n)
    for i in range(n - 1, -1, -1):
      reward = reward * discount_rate + Reward(t + i)
      value = self.policy_net.values(t + i)
      td_error = reward - value

      log_policy = self.policy_net.log_policy(t + 1)
      policy_loss += log_policy * td_error
      value_loss += tf.square(td_error)

    return policy_loss, value_loss
