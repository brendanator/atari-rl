import tensorflow as tf
import math
import numpy as np

from agents.replay_priorities import ProportionalPriorities
from .mock import Mock


class ReplayPrioritiesTest(tf.test.TestCase):
  def test_proportional_priority(self):
    config = Mock(
        replay_capacity=12, replay_alpha=1, replay_beta=1, num_steps=100)
    priorities = ProportionalPriorities(config)

    total = 11
    priorities.update_priorities(np.arange(total), np.arange(total))

    # Each index should be sampled in proportion to its value
    # Check that each the number for each index is within 3 standard deviations
    # of the expected value
    multiple = 10000
    number_samples = int(priorities.total_priority() * multiple)
    indices = priorities.sample_indices(number_samples)
    for i, count in enumerate(np.bincount(indices)):
      self.assertNear(i * multiple, count, 3 * math.sqrt(multiple))

    self.assertEqual(priorities.total_priority(), 55)
    self.assertEqual(priorities.max_priority(), 10)

    priorities.update_priorities(np.array([10]), np.array(20))
    self.assertEqual(priorities.total_priority(), 65)
    self.assertEqual(priorities.max_priority(), 20)

    priorities.update_priorities(np.array([10]), np.array(0))
    self.assertEqual(priorities.max_priority(), 9)
    self.assertEqual(priorities.total_priority(), 45)

    priorities.update_priorities(np.array([5]), np.array(12))
    self.assertEqual(priorities.max_priority(), 12)
    self.assertEqual(priorities.total_priority(), 52)

    priorities.update_to_highest_priority(4)
    self.assertEqual(priorities.max_priority(), 12)
    self.assertEqual(priorities.total_priority(), 60)


if __name__ == "__main__":
  tf.test.main()
