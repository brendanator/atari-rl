import numpy as np


class UniformPriorities(object):
  """Each transition has equal priority"""

  def __init__(self):
    pass

  def update_to_highest_priority(self, index):
    pass

  def update_priorities(self, indices, priorities):
    pass

  def sample_index(self, count):
    return np.random.randint(count)

  def probabilities(self, indices):
    return np.ones_like(indices)


class ProportionalPriorities(object):
  """Track the priorities of each transition proportional to the TD-error

  Contains a sum tree and a max tree for tracking values needed
  Each tree is implemented with an np.array for efficiency"""

  def __init__(self, config):
    self.capacity = config.replay_capacity
    self.alpha = config.replay_alpha

    self.sum_tree = np.zeros(2 * self.capacity - 1, dtype=np.float)
    self.max_tree = np.zeros(2 * self.capacity - 1, dtype=np.float)

  def total_priority(self):
    return self.sum_tree[0]

  def max_priority(self):
    return self.max_tree[0] or 1  # Default priority if tree is empty

  def update_to_highest_priority(self, leaf_index):
    self.update_scaled_priority(leaf_index, self.max_priority())

  def update_priorities(self, indices, priorities):
    priorities = np.absolute(priorities)
    for index, priority in zip(indices, priorities):
      self.update_priority(index, priority)

  def update_priority(self, leaf_index, priority):
    scaled_priority = priority**self.alpha
    self.update_scaled_priority(leaf_index, scaled_priority)

  def update_scaled_priority(self, leaf_index, scaled_priority):
    index = leaf_index + (self.capacity - 1)  # Skip the sum nodes

    self.sum_tree[index] = scaled_priority
    self.max_tree[index] = scaled_priority

    self.update_parent_priorities(index)

  def update_parent_priorities(self, index):
    parent = self.parent(index)
    sibling = self.sibling(index)

    self.sum_tree[parent] = self.sum_tree[index] + self.sum_tree[sibling]
    self.max_tree[parent] = max(self.max_tree[index], self.max_tree[sibling])

    if parent > 0:
      self.update_parent_priorities(parent)

  def sample_index(self, count):
    sample_value = np.random.random() * self.total_priority()
    return self.index_of_value(sample_value)

  def index_of_value(self, value):
    index = 0
    while True:
      if self.is_leaf(index):
        return index - (self.capacity - 1)

      left_index = self.left_child(index)
      left_value = self.sum_tree[left_index]
      if value <= left_value:
        index = left_index
      else:
        index = self.right_child(index)
        value -= left_value

  def probabilities(self, indices):
    return self.sum_tree[indices + (self.capacity - 1)]

  def is_leaf(self, index):
    return index >= self.capacity - 1

  def parent(self, index):
    return (index - 1) // 2

  def sibling(self, index):
    if index % 2 == 0:
      return index - 1
    else:
      return index + 1

  def left_child(self, index):
    return (index * 2) + 1

  def right_child(self, index):
    return (index * 2) + 2

  def __str__(self):
    sum_tree, max_tree = '', ''
    index = 0
    while True:
      end_index = index * 2 + 1
      sum_tree += str(self.sum_tree[index:end_index]) + '\n'
      max_tree += str(self.max_tree[index:end_index]) + '\n'

      if index >= self.capacity: break
      index = end_index

    return ('Sum\n' + sum_tree + '\nMax\n' + max_tree).strip()
