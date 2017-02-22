import tensorflow as tf
import math

from agents.replay_memory import LinearReplayMemory, ProportionalPriorities


class ReplayMemoryTest(tf.test.TestCase):
  def test_replay_memory(self):
    class Config(object):
      pass

    config = Config()
    config.replay_capacity = 100
    config.optimality_tightening = True
    config.optimality_tightening_steps = 4
    config.discount_rate = 0.99
    config.input_frames = 1
    config.input_shape = [1, 1]
    config.replay_prioritized = False
    config.num_bootstrap_heads = 1
    config.bootstrap_mask_probability = 1.0
    memory = LinearReplayMemory(config)

    for i in range(10):
      memory.store(i, i, i, False)
    memory.store(10, 10, 10, True)
    for i in range(10):
      memory.store(i, i, i, False)

    batch = memory.sample_batch(11, 0)

    # Test near start
    i = 3
    self.assertEqual(batch.alives[i], True)

    self.assertAllEqual(batch.past_observations[i].flatten(), [2, 1, 0, 0])
    self.assertEqual(batch.observations[i], 3)
    self.assertEqual(batch.next_observations[i], 4)
    self.assertAllEqual(batch.future_observations[i].flatten(), [5, 6, 7, 8])

    self.assertAllEqual(batch.past_actions[i].flatten(), [2, 1, 0, 0])
    self.assertEqual(batch.actions[i], 3)

    self.assertAllEqual(batch.past_rewards[i].flatten(), [2, 1, 0, 0])
    self.assertEqual(batch.rewards[i], 3)
    self.assertAllEqual(batch.future_rewards[i].flatten(), [4, 5, 6, 7])
    total_reward = sum([
        reward * config.discount_rate**(reward - 3) for reward in range(3, 11)
    ])
    self.assertNear(batch.total_rewards[i], total_reward, err=0.000001)

  def test_proportional_priority(self):
    priority = ProportionalPriorities(11, 1, 0, 1)
    for i in range(11):
      priority.update(i, i)

    # Each index should be sampled in proportion to its value
    # Check that each the number for each index is within 3 standard deviations
    # of the expected value
    multiple = 10000
    number_samples = int(priority.total_priority() * multiple)
    indices = [priority.sample_index() for i in range(number_samples)]
    for i in range(11):
      count = len([index for index in indices if index == i])
      self.assertNear(i * multiple, count, 3 * math.sqrt(multiple))

    self.assertEqual(priority.total_priority(), 55)
    self.assertEqual(priority.max_priority(), 10)

    priority.update(10, 20)
    self.assertEqual(priority.total_priority(), 65)
    self.assertEqual(priority.max_priority(), 20)

    priority.update(10, 0)
    self.assertEqual(priority.max_priority(), 9)
    self.assertEqual(priority.total_priority(), 45)

    priority.update(5, 12)
    self.assertEqual(priority.max_priority(), 12)
    self.assertEqual(priority.total_priority(), 52)


if __name__ == "__main__":
  tf.test.main()
