from . import dqn, inputs


class NetworkFactory(object):
  def __init__(self, reward_scaling, config):
    self.reward_scaling = reward_scaling
    self.config = config
    self.network_inputs = {}
    self.policy_nets = {}
    self.target_nets = {}

  def inputs(self, t):
    if t not in self.network_inputs:
      self.network_inputs[t] = inputs.Inputs(t, self.config)

    return self.network_inputs[t]

  def policy_network(self, t=0):
    if t not in self.policy_nets:
      self.policy_nets[t] = dqn.PolicyNetwork(
          inputs=self.inputs(t),
          reward_scaling=self.reward_scaling,
          config=self.config,
          reuse=len(self.policy_nets))

    return self.policy_nets[t]

  def target_network(self, t=0):
    if t not in self.target_nets:
      self.target_nets[t] = dqn.TargetNetwork(
          inputs=self.inputs(t),
          reward_scaling=self.reward_scaling,
          config=self.config,
          reuse=len(self.target_nets))

    return self.target_nets[t]

  def reset_target_network(self):
    pass
