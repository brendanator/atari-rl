import os
import tensorflow as tf
from threading import Thread

from networks.factory import NetworkFactory
import util


class Trainer(object):
  def __init__(self, config):
    util.log('Creating network and training operations')
    self.config = config

    # Creating networks
    factory = NetworkFactory(config)
    self.global_step, self.train_op = factory.create_train_ops()
    self.reset_op = factory.create_reset_target_network_op()
    self.agents = factory.create_agents()
    self.summary = factory.create_summary()

  def train(self):
    self.training = True

    util.log('Creating session and loading checkpoint')
    session = tf.train.MonitoredTrainingSession(
        checkpoint_dir=self.config.run_dir,
        save_summaries_steps=0,  # Summaries will be saved with train_op only
        config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

    with session:
      if len(self.agents) == 1:
        self.train_agent(session, self.agents[0])
      else:
        self.train_threaded(session)

    util.log('Training complete')

  def train_threaded(self, session):
    threads = []
    for i, agent in enumerate(self.agents):
      thread = Thread(target=self.train_agent, args=(session, agent))
      thread.name = 'Agent-%d' % (i + 1)
      thread.start()
      threads.append(thread)

    for thread in threads:
      thread.join()

  def train_agent(self, session, agent):
    # Populate replay memory
    if self.config.load_replay_memory:
      util.log('Loading replay memory')
      agent.replay_memory.load()
    else:
      util.log('Populating replay memory')
      agent.populate_replay_memory()

    # Initialize step counters
    step, steps_until_train = 0, self.config.train_period

    util.log('Starting training')
    while self.training and step < self.config.num_steps:
      # Start new episode
      observation, _, done = agent.new_game()

      # Play until losing
      while not done:
        self.reset_target_network(session, step)
        action = agent.action(session, step, observation)
        observation, _, done = agent.take_action(action)
        step += 1
        steps_until_train -= 1
        if done or (steps_until_train == 0):
          step = self.train_batch(session, agent.replay_memory, step)
          steps_until_train = self.config.train_period

      # Log episode
      agent.log_episode(step)

    if self.config.save_replay_memory:
      agent.replay_memory.save()

  def reset_target_network(self, session, step):
    if (self.reset_op and step > 0
        and step % self.config.target_network_update_period == 0):
      session.run(self.reset_op)

  def train_batch(self, session, replay_memory, step):
    fetches = [self.global_step, self.train_op] + self.summary.operation(step)

    batch = replay_memory.sample_batch(fetches, self.config.batch_size)
    if batch:
      step, priorities, summary = session.run(fetches, batch.feed_dict())
      batch.update_priorities(priorities)
      self.summary.add_summary(summary, step)

    return step

  def stop_training(self):
    util.log('Stopping training')
    self.training = False
