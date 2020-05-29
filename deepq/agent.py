import numpy as np
import torch
from gym import logger
from gym.spaces import Discrete

from deepq.network import DeepQNetwork
from deepq.replay_memory import ReplayMemory
from plot.plot import get_timestamp


class Agent:
    def __init__(self, gamma=0.99, epsilon=1.0, epsilon_decay=1e-5, epsilon_end=0.1, learning_rate=1e-4,
                 memory_capacity=50_000, replace_every=None, action_space=Discrete(4), state_shape=(4, 84, 84)):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate

        self.action_space = action_space

        self.replay_memory = ReplayMemory(memory_capacity, state_shape)

        self.replace_every = replace_every  # The rate at which we swap the two networks

        self.step = 0
        self.learning_step = 0

        self.q = DeepQNetwork(learning_rate, state_shape, action_space.n)
        self.next_q = DeepQNetwork(learning_rate, state_shape, action_space.n)
        self.device = self.q.device

    def act(self, observation):
        """Act in an epsilon-greedy manner.
        :param observation: nd-array representing a stack of emulator frames
        :return action: an action in the action space
        """
        if np.random.random() < 1 - self.epsilon:
            state = torch.tensor([observation], dtype=torch.float).to(self.device)
            q_values = self.q.forward(state)
            action = torch.argmax(q_values).item()
        else:
            action = self.action_space.sample()

        return action

    def learn(self, batch_size=32):
        self.q.optimizer.zero_grad()
        self._replace_target_network()

        # Sample a mini-batch from the replay memory
        states, actions, rewards, next_states, dones = self._sample_batch(batch_size)

        q_pred = self.q(states)
        q_pred = q_pred[np.arange(batch_size), actions.tolist()]

        q_next = self.next_q(next_states)
        q_next = q_next.max(dim=1)
        q_next = q_next.values

        q_next[dones] = 0.0

        q_target = rewards + self.gamma * q_next

        loss = self.q.loss(q_pred, q_target).to(self.device)
        loss.backward()
        self.q.optimizer.step()

        self.learning_step += 1
        self._decay_epsilon()

    def _replace_target_network(self):
        """Syncs the target network with the main Q-network network."""
        if self.replace_every is None:
            return

        if self.learning_step % self.replace_every == 0:
            self.next_q.load_state_dict(self.q.state_dict())

    def _sample_batch(self, batch_size):
        """Sample the batch then convert from NumPy arrays to Torch tensors."""
        arrays = self.replay_memory.sample(batch_size)
        tensors = [torch.tensor(x).to(self.device) for x in arrays]
        return tuple(tensors)

    def _decay_epsilon(self):
        new_epsilon = self.epsilon - self.epsilon_decay
        if new_epsilon <= self.epsilon_end:
            self.epsilon = self.epsilon_end
        self.epsilon = new_epsilon

    def store(self, observation, action, reward, next_observation, done):
        self.replay_memory.add_transition(observation, action, reward, next_observation, done)

    def save_checkpoint(self, path):
        path = '%s/checkpoint_from_%s.pyt' % (path, get_timestamp())
        torch.save({
            'learning_step': self.learning_step,
            'q': self.q.state_dict(),
            'next_q': self.next_q.state_dict(),
            'optimizer': self.q.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, path)
        logger.info('Saved checkpoint at %s.', path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.q.load_state_dict(checkpoint['q'])
        self.next_q.load_state_dict(checkpoint['next_q'])
        self.q.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.learning_step = checkpoint['learning_step']

        logger.info('Loaded checkpoint from %s.', path)

        self.q.train()
        self.next_q.train()
