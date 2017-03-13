import tensorflow as tf
from . import dqn, inputs, loss, reward_scaling
from agents import Agent, ReplayMemory
import util


class NetworkFactory(object):
  def __init__(self, config):
    self.config = config
    if config.reward_scaling:
      self.reward_scaling = reward_scaling.RewardScaling(config)
    else:
      self.reward_scaling = reward_scaling.DisabledRewardScaling()
    self.inputs = inputs.Inputs(config)
    self.policy_nets = {}
    self.target_nets = {}
    self.summary = util.Summary(config)

    with tf.variable_scope('policy_variables') as self.policy_scope:
      pass

    with tf.variable_scope('target_variables') as self.target_scope:
      pass

    with tf.name_scope('networks') as self.network_scope:
      pass

  def policy_network(self, t=0):
    if t not in self.policy_nets:
      reuse = len(self.policy_nets) > 0
      with tf.variable_scope(self.policy_scope, reuse=reuse) as scope:
        with tf.name_scope(self.network_scope):
          with tf.name_scope(util.format_offset('policy', t)):
            self.policy_nets[t] = dqn.Network(
                variable_scope=scope,
                inputs=self.inputs.offset_input(t),
                reward_scaling=self.reward_scaling,
                config=self.config,
                write_summaries=(t == 0))

    return self.policy_nets[t]

  def target_network(self, t=0):
    if t not in self.target_nets:
      reuse = len(self.target_nets) > 0
      with tf.variable_scope(self.target_scope, reuse=reuse) as scope:
        with tf.name_scope(self.network_scope):
          with tf.name_scope(util.format_offset('target', t)):
            self.target_nets[t] = dqn.Network(
                variable_scope=scope,
                inputs=self.inputs.offset_input(t),
                reward_scaling=self.reward_scaling,
                config=self.config,
                write_summaries=False)

    return self.target_nets[t]

  def create_agents(self):
    agents = []
    for _ in range(self.config.num_threads):
      memory = ReplayMemory(self.config)
      agent = Agent(self.policy_network(), memory, self.summary, self.config)
      agents.append(agent)

    return agents

  def create_train_ops(self):
    # Optimizer
    # optimizer = tf.train.RMSPropOptimizer(
    #     learning_rate=0.0025, momentum=0.95, epsilon=0.0001)
    # optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    optimizer = tf.train.AdamOptimizer(epsilon=0.01)

    # Create loss
    losses = loss.Losses(self, self.config)

    # Compute gradients
    policy_vars = self.policy_network().variables
    reward_scaling_vars = self.reward_scaling.variables
    trainable_vars = policy_vars + reward_scaling_vars
    grads = optimizer.compute_gradients(losses.loss, var_list=trainable_vars)

    # Apply normalized SGD for reward scaling
    grads = self.reward_scaling.scale_gradients(grads, policy_vars)

    # Clip gradients
    if self.config.grad_clipping:
      with tf.name_scope('clip_gradients'):
        grads = [(tf.clip_by_value(grad, -self.config.grad_clipping,
                                   self.config.grad_clipping), var)
                 for grad, var in grads if grad is not None]

    # Create training op
    global_step = self.inputs.global_step
    minimize = optimizer.apply_gradients(grads, global_step, name='minimize')
    with tf.control_dependencies([minimize]):
      train_op = tf.identity(losses.priorities, name='train')

    self.create_summary_ops(losses.loss, trainable_vars, grads)

    return global_step, train_op

  def create_summary_ops(self, loss, variables, gradients):
    tf.summary.scalar('loss', loss)

    for var in variables:
      tf.summary.histogram(var.name, var)

    for grad, var in gradients:
      if grad is not None:
        tf.summary.histogram('gradient/' + var.name, grad)

    self.summary.create_summary_op()

  def create_summary(self):
    return self.summary

  def create_reset_target_network_op(self):
    if self.policy_nets and self.target_nets:
      policy_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                           self.policy_scope.name)
      target_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                           self.target_scope.name)

      with tf.name_scope('reset_target_network'):
        copy_ops = []
        for from_var, to_var in zip(policy_variables, target_variables):
          name = 'reset_' + to_var.name.split('/', 1)[1][:-2].replace('/', '_')
          copy_ops.append(tf.assign(to_var, from_var, name=name))
        return tf.group(*copy_ops)
    else:
      return None
