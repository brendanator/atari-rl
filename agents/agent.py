import tensorflow as tf
import numpy as np

from atari.atari import Atari
from agents import dqn, util
from agents.optimality_tightening import ConstraintNetwork
from agents.replay_memory import ReplayMemory
from agents.exploration_bonus import ExplorationBonus
import tensorflow as tf


class Agent:
  def __init__(self, config):
    self.config = config

    # Create environment
    self.atari = Atari(config)
    self.exploration_bonus = ExplorationBonus(config)

    # Create action-value network
    self.policy_network = dqn.PolicyNetwork(config)

    # Create target action-value network
    self.target_network = dqn.TargetNetwork(self.policy_network, config)

    # Create operation to copy variables from policy network to target network
    self.reset_target_network_op = self.policy_network.copy_to_network(
        self.target_network)

    # Build loss
    self.build_loss()

    # Create training operation
    self.build_train_op()

    # Summary operation
    self.summary_op = tf.summary.merge_all()

    # Initialize replay memory
    self.replay_memory = ReplayMemory(config)
    self.prepopulate_replay_memory()

  def build_loss(self):
    # TD-errors are calculated per bootstrap head
    td_errors = (self.target_network.heads_target_action_value -
                 self.policy_network.heads_taken_action_value)

    if self.config.persistent_advantage_learning:
      self.advantage_learning_net = dqn.TargetNetwork(
          self.policy_network, self.config, reuse=True)
      self.next_state_advantage_learning_net = dqn.TargetNetwork(
          self.policy_network, self.config, reuse=True)

      advantage_learning = td_errors - self.config.pal_alpha * (
          self.advantage_learning_net.heads_max_value -
          self.advantage_learning_net.heads_taken_action_value)

      next_state_advantage_learning = td_errors - self.config.pal_alpha * (
          self.next_state_advantage_learning_net.heads_max_value -
          self.next_state_advantage_learning_net.heads_taken_action_value)

      persistent_advantage_learning = tf.maximum(
          advantage_learning,
          next_state_advantage_learning,
          name='persistent_advantage_learning')

      td_errors = persistent_advantage_learning

    # Square error are also calculated per bootstrap head
    square_errors = tf.square(td_errors)

    if self.config.optimality_tightening:
      self.constraint_network = ConstraintNetwork(self.policy_network,
                                                  self.config)
      penalty = self.constraint_network.violation_penalty
      error_rescaling = self.constraint_network.error_rescaling
      square_errors = (square_errors + penalty) / error_rescaling

    # Sum bootstrap heads
    self.td_errors = tf.reduce_sum(td_errors, axis=1, name='td_errors')
    self.error_weights = tf.placeholder(tf.float32, [None], 'error_weights')
    self.loss = tf.reduce_mean(
        self.error_weights * tf.reduce_sum(
            square_errors, axis=1), name='loss')

    if self.config.loss_clipping > 0:
      self.loss = tf.maximum(
          -self.config.loss_clipping,
          tf.minimum(self.loss, self.config.loss_clipping),
          name='loss')

  def build_train_op(self):
    # Create global step
    global_step = tf.contrib.framework.get_or_create_global_step()

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = util.add_loss_summaries(self.loss)

    # Optimizer
    opt = tf.train.AdamOptimizer()

    # Minimize loss
    with tf.control_dependencies([loss_averages_op]):
      grads = opt.compute_gradients(
          self.loss, var_list=self.policy_network.variables())

      if self.config.grad_clipping:
        grads = [(tf.clip_by_value(grad, -self.config.grad_clipping,
                                   self.config.grad_clipping), var)
                 for grad, var in grads if grad is not None]

      self.train_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
      tf.summary.histogram('trainable', var)

    # Add histograms for gradients.
    for grad, var in grads:
      if grad is not None:
        tf.summary.histogram('gradient', grad)

  def reset_target_network(self, session):
    session.run(self.reset_target_network_op)

  def new_game(self):
    self.policy_network.sample_head()
    frames, reward, done = self.atari.reset()
    self.observation = self.process_frames(frames)
    return self.observation, reward, done

  def action(self, observation, step, session):
    # Epsilon greedy exploration/exploitation even for bootstrapped DQN
    if np.random.rand() < self.epsilon(step):
      return self.atari.sample_action()
    else:
      return session.run(self.policy_network.max_action,
                         {self.policy_network.input_frames: [observation]})

  def take_action(self, action):
    frames, reward, done = self.atari.step(action)
    training_reward = self.process_reward(reward, frames)

    # Store action, reward and done with the last observation
    self.replay_memory.store(self.observation, action, training_reward, done)

    self.observation = self.process_frames(frames)
    return self.observation, reward, done

  def process_frames(self, frames):
    observation = []
    for i in range(-self.config.input_frames, 0):
      image = util.process_image(frames[i - 1], frames[i],
                                 self.config.input_shape)
      image /= 255.0  # Normalize each pixel between 0 and 1
      observation.append(image)
    return observation

  def process_reward(self, reward, frames):
    if self.config.exploration_bonus:
      reward += self.exploration_bonus.bonus(frames)

    if self.config.reward_clipping:
      reward = max(-self.config.reward_clipping,
                   min(reward, self.config.reward_clipping))

    return reward

  def epsilon(self, step):
    """Epsilon is linearly annealed from an initial exploration value
    to a final exploration value over a number of steps"""

    initial = self.config.initial_exploration
    final = self.config.final_exploration
    final_frame = self.config.final_exploration_frame

    annealing_rate = (initial - final) / final_frame
    annealed_exploration = initial - (step * annealing_rate)
    return max(annealed_exploration, final)

  def prepopulate_replay_memory(self):
    """Play game with random actions to populate the replay memory"""

    count = 0
    done = True

    while count <= self.config.replay_start_size or not done:
      if done: self.new_game()
      _, _, done = self.take_action(self.atari.sample_action())
      count += 1

  def train(self, session, step, summary=False):
    batch = self.replay_memory.sample_batch(self.config.batch_size, step)
    feed_dict = self.build_feed_dict(batch)

    if summary:
      td_errors, _, summary = session.run(
          [self.td_errors, self.train_op, self.summary_op], feed_dict)
    else:
      td_errors, _ = session.run([self.td_errors, self.train_op], feed_dict)

    self.replay_memory.update_priorities(batch.indices, td_errors)

    return summary

  def build_feed_dict(self, batch):
    feed_dict = {
        self.policy_network.input_frames: batch.observations,
        self.policy_network.action_input: batch.actions,
        self.target_network.reward_input: batch.rewards,
        self.target_network.alive_input: batch.alives,
        self.target_network.input_frames: batch.next_observations,
        self.error_weights: batch.error_weights
    }

    if self.config.optimality_tightening:
      constraint_feed_dict = {
          self.constraint_network.past_input_frames: batch.past_observations,
          self.constraint_network.past_actions: batch.past_actions,
          self.constraint_network.past_rewards: batch.past_rewards,
          self.constraint_network.past_alives: batch.past_alives,
          self.constraint_network.rewards: batch.rewards,
          self.constraint_network.alives: batch.alives,
          self.constraint_network.future_input_frames:
          batch.future_observations,
          self.constraint_network.future_rewards: batch.future_rewards,
          self.constraint_network.future_alives: batch.future_alives,
          self.constraint_network.total_rewards: batch.total_rewards
      }
      feed_dict.update(constraint_feed_dict)

    if self.config.bootstrapped and self.config.bootstrap_mask_probability < 1.0:
      feed_dict[self.policy_network.bootstrap_mask] = batch.bootstrap_mask

    if self.config.persistent_advantage_learning:
      persistent_advantage_feed_dict = {
          self.advantage_learning_net.input_frames: batch.observations,
          self.advantage_learning_net.action_input: batch.actions,
          self.next_state_advantage_learning_net.input_frames:
          batch.next_observations,
          self.next_state_advantage_learning_net.action_input: batch.actions
      }
      feed_dict.update(persistent_advantage_feed_dict)

    return feed_dict
