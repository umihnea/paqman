from dataclasses import dataclass

import torch
import numpy as np


@dataclass
class Batch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray

    def as_tensors(self, device):
        """Sample the batch then convert from NumPy arrays to Torch tensors.
        """
        states = torch.from_numpy(self.states).float().to(device)
        next_states = torch.from_numpy(self.next_states).float().to(device)
        actions = torch.from_numpy(self.actions).to(device)
        rewards = torch.from_numpy(self.rewards).float().to(device)
        dones = torch.from_numpy(self.dones).bool().to(device)

        return states, actions, rewards, next_states, dones
