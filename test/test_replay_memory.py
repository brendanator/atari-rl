import numpy as np
import tensorflow as tf

from agents.replay_memory import ReplayMemory, SampleBatch
from networks.inputs import Inputs
from .mock import Mock


class ReplayMemoryTest(tf.test.TestCase):
  def test_replay_memory(self):
    config = Mock(
        replay_capacity=15,
        discount_rate=0.99,
        input_frames=3,
        input_shape=[],
        replay_priorities='uniform',
        num_bootstrap_heads=1,
        bootstrap_mask_probability=1.0,
        run_dir='',
        async=True)
    memory = ReplayMemory(config)

    memory.store_new_episode([0, 1])
    for i in range(2, 11):
      memory.store_transition(i - 1, i - 1, False, [i - 1, i])
    memory.store_transition(10, 10, True, [10, 11])

    inputs = Inputs(config)
    fetches = [
        inputs.offset_input(0).frames,
        inputs.offset_input(-1).frames,
        inputs.offset_input(0).action,
        inputs.offset_input(1).reward,
        inputs.offset_input(1).alive,
        inputs.offset_input(2).alive,
        inputs.offset_input(0).discounted_reward,
    ]
    batch = memory.sample_batch(fetches, batch_size=2)
    feed_dict = batch.feed_dict()

    self.assertAllEqual(batch.indices, [4, 9])

    # The 4 values come from t=0 and t=-1 with input_frames=3
    self.assertAllEqual(feed_dict[inputs.frames], [[1, 2, 3, 4], [6, 7, 8, 9]])
    self.assertAllEqual(feed_dict[inputs.actions], [[4], [9]])
    self.assertAllEqual(feed_dict[inputs.rewards], [[5], [10]])
    self.assertAllEqual(feed_dict[inputs.alives],
                        [[True, True], [True, False]])

    discounted_reward = sum([
        reward * config.discount_rate**(reward - 4) for reward in range(4, 11)
    ])
    self.assertNear(
        feed_dict[inputs.discounted_rewards][0], discounted_reward, err=0.0001)


if __name__ == "__main__":
  tf.test.main()
