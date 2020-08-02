import numpy as np

from common.sum_tree import SumTree
from replay_memory.batch import Batch
from replay_memory.replay_memory import ReplayMemory

import logging


class PrioritizedReplayMemory(ReplayMemory):
    def __init__(self, raw_space, state_shape, alpha):
        super().__init__(raw_space, state_shape)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)

        # Hyperparameters
        self.alpha = alpha
        self.max_priority = 1.0

        # Find tree capacity from raw linear capacity
        tree_capacity = 1
        while tree_capacity < self.capacity:
            tree_capacity *= 2
        self.capacity = tree_capacity

        self.tree = SumTree(tree_capacity)
        self.logger.debug("Tree initialized with capacity {}.", tree_capacity)

    def add_transition(self, state, action, reward, next_state, done):
        super().add_transition(state, action, reward, next_state, done)
        initial_priority = self.max_priority ** self.alpha
        self.tree[self.size - 1] = initial_priority
        self.logger.debug("Added to tree with initial priority {}.", initial_priority)

    def _sample_indices(self, batch_size):
        indices = []

        total_priority = self.tree.total
        segment_length = total_priority / batch_size

        for i in range(batch_size):
            mass = np.random.uniform(i * segment_length, (i + 1) * segment_length)
            index = self.tree.find_prefix_sum(mass)
            indices.append(index)

        return indices

    def sample(self, batch_size, beta=None):
        indices = self._sample_indices(batch_size)
        weigths = []
        # todo: to be contd
