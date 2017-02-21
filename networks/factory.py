from . import dqn, inputs
from .reward_scaling import *


class NetworkFactory(object):
  def __init__(self, config):
    self.config = config
    if config.reward_scaling:
      self.reward_scaling = RewardScaling(config)
    else:
      self.reward_scaling = DisabledRewardScaling()
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

  def create_reset_target_network_op(self):
    if self.target_nets:
      policy_network = self.policy_nets.popitem()[1]
      target_network = self.target_nets.popitem()[1]
      self.reset_op = policy_network.copy_to_network(target_network)
    else:
      self.reset_op = None

  def reset_target_network(self, session, step):
    if self.reset_op and step % self.config.target_network_update_period == 0:
      session.run(self.reset_op)
