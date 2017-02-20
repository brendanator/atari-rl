class BaseTrainer(object):
  def train_step(self, session, step, summary):
    batch = self.replay_memory.sample_batch(self.config.batch_size, step)
    feed_dict = self.build_feed_dict(batch)

    if summary:
      global_step, td_errors, _, summary = session.run(
          [self.global_step, self.td_errors, self.train_op, self.summary_op],
          feed_dict)
      self.summary_writer.add_summary(summary, step)
    else:
      global_step, td_errors, _ = session.run(
          [self.global_step, self.td_errors, self.train_op], feed_dict)

    self.replay_memory.update_priorities(batch.indices, td_errors)

    return global_step
