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
    self.global_inputs = {}
    self.network_inputs = {}
    self.policy_nets = {}
    self.target_nets = {}

  def global_input(self, scope):
    if scope not in self.global_inputs:
      self.global_inputs[scope] = inputs.GlobalInputs(self.config)

    return self.global_inputs[scope]

  def inputs(self, scope, t):
    key = (scope, t)
    if key not in self.network_inputs:
      self.network_inputs[key] = inputs.NetworkInputs(t, self.config)

    return self.network_inputs[key]

  def policy_network(self, scope, t=0):
    key = (scope, t)
    if key not in self.policy_nets:
      self.policy_nets[key] = dqn.PolicyNetwork(
          inputs=self.inputs(scope, t),
          reward_scaling=self.reward_scaling,
          config=self.config,
          reuse=len(self.policy_nets) > 0)

    return self.policy_nets[key]

  def target_network(self, scope, t=0):
    key = (scope, t)
    if key not in self.target_nets:
      self.target_nets[key] = dqn.TargetNetwork(
          inputs=self.inputs(scope, t),
          reward_scaling=self.reward_scaling,
          config=self.config,
          reuse=len(self.target_nets) > 0)

    return self.target_nets[key]

  def create_agents(self):
    agents = []
    with tf.device('/gpu:0'):
      global_step, train_ops = self.create_train_ops()
    offsets = [t for (_, t) in self.network_inputs.keys()]

    for i in range(self.config.num_threads):
      policy_network = self.policy_network(self.gpu_scope(i))
      train_op = train_ops[i % self.config.num_gpus]
      memory = ReplayMemory(offsets, self.config)
      agent = Agent(policy_network, train_op, memory, self.config)
      agents.append(agent)

    return global_step, agents

  def create_train_ops(self):
    # Optimizer
    optimizer = tf.train.AdamOptimizer()

    train_ops = []
    with tf.variable_scope(tf.get_variable_scope()):
      for i in range(self.config.num_gpus):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope(self.gpu_scope(i)) as scope:
            td_error, grads = self.simple_gradients(optimizer, scope)

            # Clip gradients
            if self.config.grad_clipping:
              grads = [(tf.clip_by_value(grad, -self.config.grad_clipping,
                                         self.config.grad_clipping), var)
                       for grad, var in grads if grad is not None]

            # Create training op
            global_step = tf.contrib.framework.get_or_create_global_step()
            minimize = optimizer.apply_gradients(
                grads, global_step=global_step)

            # with tf.control_dependencies([loss_summaries, minimize]):
            with tf.control_dependencies([minimize]):
              train_op = tf.identity(td_error, name='train')
            train_ops.append(train_op)

            # Reuse variables for the next tower
            tf.get_variable_scope().reuse_variables()

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
      tf.summary.histogram('trainable', var)

    # Add histograms for gradients.
    for grad, var in grads:
      if grad is not None:
        tf.summary.histogram('gradient', grad)

    return global_step, train_ops

  def calculate_gradients(self, optimizer):
    if self.config.num_gpus == 0:
      with tf.device('/cpu:0'):
        return self.simple_gradients(optimizer)
    elif self.config.num_gpus == 1:
      with tf.device('/gpu:0'):
        return self.simple_gradients(optimizer)
    else:
      return self.gpu_gradients(optimizer)

  def gpu_train_ops(self, optimizer):
    # gradient_queue = tf.FIFOQueue()
    gpu_train_ops = []
    with tf.variable_scope(tf.get_variable_scope()):
      for i in range(self.config.num_gpus):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope(self.gpu_scope(i)) as scope:
            td_error, gradients = self.simple_gradients(optimizer, scope)
            # gpu_gradient_op = gradient_queue.enqueue(gradients)
            with tf.control_dependencies([gpu_gradient_op]):
              gpu_train_op = tf.identity(td_error)
            gpu_train_ops.append(gpu_train_op)

            # Reuse variables for the next tower
            tf.get_variable_scope().reuse_variables()

    # TODO maybe use?
    # average_grads = []
    # for grad_and_vars in zip(*tower_grads):
    #   grad = tf.stack([grad for grad, _ in grad_and_vars], axis=0)
    #   grad = tf.reduce_mean(grad, 0)
    #   var = grad_and_vars[0][1]  # They're all the same, so just use the first
    #   average_grads.append((grad, var))

    # Retain the summaries from the final tower.
    # summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

    return gradient_queue.dequeue(), gpu_train_ops

  def gpu_scope(self, index):
    if self.config.num_gpus <= 1:
      return None
    else:
      gpu = index % self.config.num_gpus
      return '%s_%d' % (util.TOWER_NAME, index)

  def simple_gradients(self, optimizer, scope=None):
    # Calculate the loss for one tower of the model. This function constructs
    # the entire model but shares the variables across all towers.
    losses = loss.Losses(self, scope, self.config)
    loss_summaries = util.add_loss_summaries(losses.loss, scope)

    # Calculate the gradients for the batch of data on this tower.
    policy_vars = self.policy_network(scope).variables
    reward_scaling_vars = self.reward_scaling.variables
    grads = optimizer.compute_gradients(
        losses.loss, var_list=policy_vars + reward_scaling_vars)

    # Apply normalized SGD for reward scaling
    grads = self.reward_scaling.scale_gradients(grads, policy_vars)

    return losses.td_error, grads

  def create_reset_target_network_op(self):
    if self.policy_nets and self.target_nets:
      policy_network = self.random_dict_value(self.policy_nets)
      target_network = self.random_dict_value(self.target_nets)
      return policy_network.copy_to_network(target_network)
    else:
      return None

  def random_dict_value(self, dict):
    key, value = dict.popitem()
    dict[key] = value
    return value
