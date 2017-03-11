import tensorflow as tf


class Summary(object):
  def __init__(self, config):
    self.summary_writer = tf.summary.FileWriter(config.run_dir)
    self.summary_step_period = config.summary_step_period

  def run_summary(self, step):
    return step % self.summary_step_period == 0

  def epsilon(self, step, epsilon):
    if self.run_summary(step):
      summary = tf.Summary()
      summary.value.add(tag='epsilon', simple_value=epsilon)
      self.summary_writer.add_summary(summary, step)

  def episode(self, step, score, steps, duration):
    summary = tf.Summary()
    summary.value.add(tag='episode/score', simple_value=score)
    summary.value.add(tag='episode/steps', simple_value=steps)
    summary.value.add(tag='episode/time', simple_value=duration)
    summary.value.add(
        tag='episode/reward_per_sec', simple_value=score / duration)
    summary.value.add(
        tag='episode/steps_per_sec', simple_value=steps / duration)
    self.summary_writer.add_summary(summary, step)

  def operation(self, step):
    if self.run_summary(step):
      return [self.summary_op]
    else:
      return [self.dummy_summary_op]

  def add_summary(self, summary, step):
    if summary:
      self.summary_writer.add_summary(summary, step)

  def create_summary_op(self):
    self.summary_op = tf.summary.merge_all()
    self.dummy_summary_op = tf.no_op(name='dummy_summary')
