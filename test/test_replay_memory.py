import tensorflow as tf
from agents.replay_memory import *


class ReplayMemoryTest(tf.test.TestCase):
  def test_extended_list(self):
    extended_list = ExtendedList(0, 1)
    extended_list.append(2)

    self.assertEqual(extended_list[-1], 0)
    self.assertEqual(extended_list[3], 2)

    self.assertEqual(extended_list[-2:4], [0, 0, 0, 1, 2, 2])

  def test_episode_memory(self):
    discount_rate = 0.99
    memory = EpisodeMemory(
        initial_observation=0, discount_rate=discount_rate, bound_steps=4)

    transitions = [memory.add_step(i, i, False, i + 1) for i in range(10)]
    transitions += [memory.add_step(10, 10, True, 11)]

    # Test near start
    transition = transitions[3]
    self.assertEqual(transition.done, False)

    self.assertEqual(transition.past_observations, [0, 0, 1, 2])
    self.assertEqual(transition.observation, 3)
    self.assertEqual(transition.next_observation, 4)
    self.assertEqual(transition.future_observations, [5, 6, 7, 8])

    self.assertEqual(transition.past_actions, [0, 0, 1, 2])
    self.assertEqual(transition.action, 3)

    first_reward = ((1 / discount_rate) + 2) / discount_rate
    second_reward = 2 / discount_rate
    self.assertEqual(transition.past_rewards,
                     [first_reward, first_reward, first_reward, second_reward])
    self.assertEqual(transition.reward, 3)
    future1_reward = 3 + 4 * discount_rate
    future2_reward = future1_reward + 5 * discount_rate**2
    future3_reward = future2_reward + 6 * discount_rate**3
    future4_reward = future3_reward + 7 * discount_rate**4
    self.assertEqual(
        transition.future_rewards,
        [future1_reward, future2_reward, future3_reward, future4_reward])

    # Test near end
    transition = transitions[9]
    self.assertEqual(transition.done, False)
    self.assertEqual(transition.future_done, [True, True, True, True])

    self.assertEqual(transition.past_observations, [5, 6, 7, 8])
    self.assertEqual(transition.observation, 9)
    self.assertEqual(transition.next_observation, 10)
    self.assertEqual(transition.future_observations, [11, 11, 11, 11])

    self.assertEqual(transition.past_actions, [5, 6, 7, 8])
    self.assertEqual(transition.action, 9)

    past_reward_4 = 8 / discount_rate
    past_reward_3 = past_reward_4 + 7 / discount_rate**2
    past_reward_2 = past_reward_3 + 6 / discount_rate**3
    past_reward_1 = past_reward_2 + 5 / discount_rate**4
    self.assertArrayNear(
        transition.past_rewards,
        [past_reward_1, past_reward_2, past_reward_3, past_reward_4], 1e-14)
    self.assertEqual(transition.reward, 9)
    future_reward = 9 + 10 * discount_rate
    self.assertEqual(transition.future_rewards, [future_reward] * 4)
    self.assertEqual(transitions[-1].total_reward, 10)
    self.assertEqual(transition.total_reward, 9 + 10 * discount_rate)

    # Test final
    transition = transitions[-1]
    self.assertEqual(transition.done, True)
    self.assertEqual(transition.total_reward, 10)


if __name__ == "__main__":
  tf.test.main()
