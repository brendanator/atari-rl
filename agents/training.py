import os
import tensorflow as tf
from threading import Thread

from networks import NetworkFactory
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

    # Summary writer
    self.summary_writer = tf.summary.FileWriter(config.run_dir)
    self.summary_op = tf.summary.merge_all()

  def train(self):
    self.training = True
    self.save_replay_memory = True

    util.log('Creating session and loading checkpoint')
    session = tf.train.MonitoredTrainingSession(
        checkpoint_dir=self.config.run_dir,
        save_summaries_steps=0  # Summaries will be saved with train_op only
    )

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
    util.log('Populating replay memory')
    agent.populate_replay_memory()

    # Initialize step counters
    global_step, start_step, step = 0, 0, 0

    util.log('Starting training')
    while self.training and global_step < self.config.num_steps:
      # Start new episode
      observation, _, done = agent.new_game()

      # Play until losing
      while not done:
        self.reset_target_network(session, step)
        action = agent.action(session, step, observation)
        observation, _, done = agent.take_action(action)
        step += 1
        if done or (step - start_step == self.config.train_period):
          global_step = self.train_batch(session, agent.replay_memory,
                                         global_step)
          start_step = step

      # Log episode
      agent.log_episode(self.summary_writer, global_step)

    if self.save_replay_memory:
      agent.replay_memory.save()

  def reset_target_network(self, session, step):
    if self.reset_op and step % self.config.target_network_update_period == 0:
      session.run(self.reset_op)

  def train_batch(self, session, replay_memory, global_step):
    batch = replay_memory.sample_batch(self.config.batch_size, global_step)
    if not batch.is_valid:
      return global_step

    if global_step > 0 and global_step % self.config.summary_step_period == 0:
      fetches = [self.global_step, self.train_op, self.summary_op]
      feed_dict = batch.build_feed_dict(fetches)
      global_step, priorities, summary = session.run(fetches, feed_dict)
      self.summary_writer.add_summary(summary, global_step)
    else:
      fetches = [self.global_step, self.train_op]
      feed_dict = batch.build_feed_dict(fetches)
      global_step, priorities = session.run(fetches, feed_dict)

    batch.update_priorities(priorities)

    return global_step

  def stop_training(self, save_replay_memory):
    util.log('Stopping training')
    self.training = False
    self.save_replay_memory = save_replay_memory
