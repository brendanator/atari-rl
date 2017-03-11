import numpy as np


class UniformPriorities(object):
  """Each transition has equal priority"""

  def __init__(self):
    self.num_values = 0

  def update_to_highest_priority(self, index):
    self.num_values = max(self.num_values, index)

  def update_priorities(self, indices, priorities):
    pass

  def sample_indices(self, count):
    return np.random.randint(self.num_values, size=count)

  def probabilities(self, indices):
    return np.ones_like(indices)


class ProportionalPriorities(object):
  """Track the priorities of each transition proportional to the TD-error

  Contains a sum tree and a max tree for tracking values needed
  Each tree is implemented with an np.array for efficiency"""

  def __init__(self, config):
    # Set capacity to be a power of 2 so we have balanced trees
    self.depth = int(np.ceil(np.log2(config.replay_capacity)))
    self.capacity = np.power(2, self.depth)

    # First index in tree array is ignored to simplify calculations
    self.root_node = 1
    self.sum_tree = np.zeros(2 * self.capacity, dtype=np.float32)
    self.max_tree = np.zeros(2 * self.capacity, dtype=np.float32)

    self.alpha = config.replay_alpha

  def total_priority(self):
    return self.sum_tree[self.root_node]

  def max_priority(self):
    return self.max_tree[self.root_node] or 1  # Default value if tree is empty

  def update_to_highest_priority(self, leaf_index):
    self.update_scaled_priorites(leaf_index, self.max_priority())

  def update_priorities(self, indices, priorities):
    self.update_scaled_priorites(indices, np.absolute(priorities)**self.alpha)

  def update_scaled_priorites(self, indices, scaled_priorities):
    indices += self.capacity

    self.sum_tree[indices] = scaled_priorities
    self.max_tree[indices] = scaled_priorities

    for _ in range(self.depth):
      siblings = self.sibling(indices)
      parents = self.parent(indices)
      self.sum_tree[parents] = self.sum_tree[indices] + self.sum_tree[siblings]
      self.max_tree[parents] = np.maximum(self.max_tree[indices],
                                          self.max_tree[siblings])
      indices = parents

  def sample_indices(self, count):
    values = np.random.random(count) * self.total_priority()
    indices = np.ones(count, dtype=np.int32)

    for _ in range(self.depth):
      left_children = self.left_child(indices)
      left_values = self.sum_tree[left_children]
      go_right = values > self.sum_tree[left_children]
      indices = left_children + go_right
      values -= left_values * go_right

    return indices - self.capacity

  def probabilities(self, indices):
    return self.sum_tree[indices + self.capacity]

  def parent(self, index):
    return index >> 1

  def sibling(self, index):
    return index ^ 1

  def left_child(self, index):
    return index * 2

  def right_child(self, index):
    return index * 2 + 1

  def __str__(self):
    sum_tree, max_tree = '', ''
    index = self.root_node
    while True:
      end_index = self.left_child(index)
      sum_tree += str(self.sum_tree[index:end_index]) + '\n'
      max_tree += str(self.max_tree[index:end_index]) + '\n'

      if index >= self.capacity: break
      index = end_index

    return ('Sum\n' + sum_tree + '\nMax\n' + max_tree).strip()
