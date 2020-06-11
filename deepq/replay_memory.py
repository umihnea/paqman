import numpy as np


class ReplayMemory:
    def __init__(self, raw_space, state_shape):
        capacity = self.from_gigabytes(raw_space, state_shape)
        self._states = np.empty((capacity, *state_shape), np.float32)
        self._next_states = np.empty((capacity, *state_shape), np.float32)
        self._actions = np.empty(capacity, np.uint8)
        self._rewards = np.empty(capacity, np.float32)
        self._dones = np.empty(capacity, np.bool)
        self.capacity = capacity
        self.size = 0

    @staticmethod
    def from_gigabytes(gigabytes: float, state_shape) -> int:
        array_size = state_shape[0] * state_shape[1] * state_shape[2]
        return int(np.rint((gigabytes * 1.074e9) // (8 * array_size + 6)).item())

    def sample(self, batch_size):
        """Sample a mini-batch of transitions from the replay memory.
        If there are not enough transitions for a full mini-batch, we will return a smaller mini-batch.
        """
        end = min(self.size, self.capacity)
        sample_indexes = np.random.choice(end, batch_size, replace=False)
        states = self._states[sample_indexes]
        actions = self._actions[sample_indexes]
        rewards = self._rewards[sample_indexes]
        next_states = self._next_states[sample_indexes]
        dones = self._dones[sample_indexes]

        return states, actions, rewards, next_states, dones

    def add_transition(self, state, action, reward, next_state, done):
        """Place the next transition at position i % N,
        where i is the next free cell and N is the memory capacity.
        """
        pos = self.size % self.capacity
        self._states[pos] = state
        self._actions[pos] = action
        self._rewards[pos] = reward
        self._next_states[pos] = next_state
        self._dones[pos] = int(done)
        self.size += 1
