import numpy as np


def overlaps(x, y, a, b) -> bool:
    """Returns true if [x, y] is completely included within [A, B].
    """
    return a <= x <= y <= b


class SumTree:
    def __init__(self, capacity):
        assert capacity & (capacity - 1) == 0  # capacity must be a power of two
        self.capacity = capacity
        self.values = np.zeros(2 * capacity, dtype=np.int)

    def _query_recursively(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self.values[node]

        mid = (node_start + node_end) // 2
        if overlaps(start, end, node_start, mid):
            return self._query_recursively(start, end, 2 * node, node_start, mid)
        elif overlaps(start, end, mid + 1, node_end):
            return self._query_recursively(start, end, 2 * node + 1, mid + 1, node_end)
        else:
            # Query interval partially overlaps both children
            left = self._query_recursively(start, mid, 2 * node, node_start, mid)
            right = self._query_recursively(
                mid + 1, end, 2 * node + 1, mid + 1, node_end
            )
            return left + right

    def query(self, start=0, end=None):
        """Returns the sum of all elements in the interval given by start and end.
        """
        if end is None:
            end = self.capacity - 1
        return self._query_recursively(start, end, 1, 0, self.capacity - 1)

    @property
    def total(self):
        return self.query()

    def __setitem__(self, i, value):
        adjusted_i = i + self.capacity
        self.values[adjusted_i] = value

        # Propagate upwards
        parent = adjusted_i // 2
        while parent >= 1:
            self.values[parent] = self.values[2 * parent] + self.values[2 * parent + 1]
            parent //= 2

    def __getitem__(self, i):
        assert 0 <= i < self.capacity
        return self.values[i + self.capacity]

    def _recursive_prefix_sum(self, index, _sum):
        left = 2 * index
        right = left + 1

        if left >= len(self.values):
            return index

        if _sum <= self.values[left]:
            return self._recursive_prefix_sum(left, _sum)
        else:
            return self._recursive_prefix_sum(right, _sum - self.values[left])

    def find_prefix_sum(self, prefix_sum):
        """Find an index i in the tree such that the sum up until i - 1 <= prefix_sum.
        """
        return self._recursive_prefix_sum(1, prefix_sum) - self.capacity
