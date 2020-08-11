import numpy as np

from replay_memory.batch import Batch
from replay_memory.transition import Transition


class ReplayMemory:
    def __init__(self, raw_space, state_shape):
        self.memory = []
        self.capacity = self.from_gigabytes(raw_space, state_shape)
        self.size = 0

    @staticmethod
    def from_gigabytes(gigabytes: float, state_shape) -> int:
        """Approximate maximum array capacity based on a memory limit given in
        gigabytes."""
        array_size = state_shape[0] * state_shape[1] * state_shape[2]
        return int(np.rint((gigabytes * 1.074e9) // (8 * array_size + 6)).item())

    def add_transition(self, state, action, reward, next_state, done):
        """Place the next transition at position i % N, where i is the next
        free cell and N is the memory capacity."""
        transition = Transition(state, action, reward, next_state, done)
        if self.size < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.size % self.capacity] = transition

        self.size += 1

    def sample(self, batch_size) -> Batch:
        end = min(self.size, self.capacity)

        # Select transitions uniformly at random
        transitions = [
            self.memory[i] for i in np.random.choice(end, batch_size, replace=False)
        ]

        return Batch(
            np.array([t.state for t in transitions]),
            np.array([t.action for t in transitions]),
            np.array([t.reward for t in transitions]),
            np.array([t.next_state for t in transitions]),
            np.array([t.done for t in transitions]),
        )
