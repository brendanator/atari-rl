import tensorflow as tf


class Summaries(object):
  def __init__(self, run_dir):
    self.summary_writer = tf.summary.FileWriter(run_dir)

  def epsilon(self, step, epsilon):
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

  def add_summary(self, summary, step):
    self.summary_writer.add_summary(summary, step)

  def create_summary_op(self):
    self.summary_op = tf.summary.merge_all()
