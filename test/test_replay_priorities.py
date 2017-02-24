import tensorflow as tf
import math

from agents.replay_priorities import ProportionalPriorities
from .mock import Mock


class ReplayPrioritiesTest(tf.test.TestCase):
  def test_proportional_priority(self):
    config = Mock(
        replay_capacity=100, replay_alpha=1, replay_beta=1, num_steps=100)
    priorities = ProportionalPriorities(config)

    total = 11
    for i in range(total):
      priorities.update_priority(i, i)

    # Each index should be sampled in proportion to its value
    # Check that each the number for each index is within 3 standard deviations
    # of the expected value
    multiple = 10000
    number_samples = int(priorities.total_priority() * multiple)
    indices = [priorities.sample_index(total) for i in range(number_samples)]
    for i in range(total):
      count = len([index for index in indices if index == i])
      self.assertNear(i * multiple, count, 3 * math.sqrt(multiple))

    self.assertEqual(priorities.total_priority(), 55)
    self.assertEqual(priorities.max_priority(), 10)

    priorities.update_priority(10, 20)
    self.assertEqual(priorities.total_priority(), 65)
    self.assertEqual(priorities.max_priority(), 20)

    priorities.update_priority(10, 0)
    self.assertEqual(priorities.max_priority(), 9)
    self.assertEqual(priorities.total_priority(), 45)

    priorities.update_priority(5, 12)
    self.assertEqual(priorities.max_priority(), 12)
    self.assertEqual(priorities.total_priority(), 52)


if __name__ == "__main__":
  tf.test.main()
