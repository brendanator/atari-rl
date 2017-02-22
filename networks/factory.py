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
    self.global_inputs = inputs.GlobalInputs(config)
    self.network_inputs = {}
    self.policy_nets = {}
    self.target_nets = {}

  def inputs(self, t):
    if t not in self.network_inputs:
      self.network_inputs[t] = inputs.NetworkInputs(t, self.config)

    return self.network_inputs[t]

  def policy_network(self, t=0):
    if t not in self.policy_nets:
      self.policy_nets[t] = dqn.PolicyNetwork(
          inputs=self.inputs(t),
          reward_scaling=self.reward_scaling,
          config=self.config,
          reuse=len(self.policy_nets) > 0)

    return self.policy_nets[t]

  def target_network(self, t=0):
    if t not in self.target_nets:
      self.target_nets[t] = dqn.TargetNetwork(
          inputs=self.inputs(t),
          reward_scaling=self.reward_scaling,
          config=self.config,
          reuse=len(self.target_nets) > 0)

    return self.target_nets[t]

  def create_agents(self):
    agents = []
    for _ in range(self.config.num_threads):
      pre_offset = min(self.network_inputs.keys())
      post_offset = max(self.network_inputs.keys())
      memory = ReplayMemory(pre_offset, post_offset, self.config)
      agent = Agent(self.policy_network(), memory, self.config)
      agents.append(agent)

    return agents

  def create_train_ops(self):
    # Optimizer
    opt = tf.train.AdamOptimizer()

    # Create loss
    losses = loss.Losses(self, self.config)

    # Compute gradients
    policy_vars = self.policy_network().variables
    reward_scaling_vars = self.reward_scaling.variables
    grads = opt.compute_gradients(
        losses.loss, var_list=policy_vars + reward_scaling_vars)

    # Apply normalized SGD for reward scaling
    grads = self.reward_scaling.scale_gradients(grads, policy_vars)

    # Clip gradients
    if self.config.grad_clipping:
      grads = [(tf.clip_by_value(grad, -self.config.grad_clipping,
                                 self.config.grad_clipping), var)
               for grad, var in grads if grad is not None]

    # Create training op
    loss_summaries = util.add_loss_summaries(losses.loss)
    global_step = tf.contrib.framework.get_or_create_global_step()
    minimize = opt.apply_gradients(grads, global_step=global_step)
    with tf.control_dependencies([loss_summaries, minimize]):
      train_op = tf.identity(losses.td_error, name='train')

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
      tf.summary.histogram('trainable', var)

    # Add histograms for gradients.
    for grad, var in grads:
      if grad is not None:
        tf.summary.histogram('gradient', grad)

    return global_step, train_op

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
