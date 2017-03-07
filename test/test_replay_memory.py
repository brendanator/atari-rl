import tensorflow as tf

from agents.replay_memory import ReplayMemory, SampleBatch
from .mock import Mock


class ReplayMemoryTest(tf.test.TestCase):
  def test_replay_memory(self):
    config = Mock(
        replay_capacity=12,
        optimality_tightening=True,
        optimality_tightening_steps=4,
        discount_rate=0.99,
        input_frames=1,
        input_shape=[],
        replay_priorities='uniform',
        num_bootstrap_heads=1,
        bootstrap_mask_probability=1.0,
        run_dir='',
        async=None)
    pre_input_offset = -3
    post_input_offset = 2
    memory = ReplayMemory(pre_input_offset, post_input_offset, config)

    memory.store_new_episode(0)
    for i in range(1, 10):
      memory.store_transition(i - 1, i - 1, False, i)
    memory.store_transition(9, 9, True, 10)

    indices = memory.recent_indices(2)
    batch = SampleBatch(memory, indices, 0)
    self.assertAllEqual(indices, [8, 3])

    for offset in range(pre_input_offset, post_input_offset):
      self.assertAllEqual(
          batch.observations(offset), indices.reshape([-1, 1]) + offset)
      self.assertAllEqual(batch.actions(offset), indices + offset)
      self.assertAllEqual(batch.rewards(offset), indices + offset)
      self.assertAllEqual(batch.alives(offset), [True, True])

    # Final offset is a little different
    offset = post_input_offset
    self.assertAllEqual(batch.observations(offset), [[10], [5]])
    self.assertAllEqual(batch.actions(offset), [0, 5])
    self.assertAllEqual(batch.rewards(offset), [0, 5])
    self.assertAllEqual(batch.alives(offset), [False, True])

    total_reward = sum([
        reward * config.discount_rate**(reward - 3) for reward in range(3, 10)
    ])
    self.assertNear(batch.total_rewards(0)[1], total_reward, err=0.0001)


if __name__ == "__main__":
  tf.test.main()
