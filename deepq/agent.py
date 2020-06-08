import numpy as np
import torch
from gym.spaces import Discrete

from deepq.network import DeepQNetwork
from deepq.replay_memory import ReplayMemory


class Agent:
    def __init__(self, model_parameters, action_space=Discrete(4), state_shape=(4, 84, 84)):
        self.learning_rate = float(model_parameters['learning_rate'])
        self.gamma = float(model_parameters['gamma'])

        self.epsilon = float(model_parameters['epsilon'])
        self.epsilon_end = float(model_parameters['epsilon_end'])
        self.epsilon_decay = float(model_parameters['epsilon_decay'])
        self.batch_size = int(model_parameters['batch_size']) or 32
        self.replace_every = int(model_parameters['replace_every'])

        self.action_space = action_space

        self.replay_memory = ReplayMemory(float(model_parameters['memory_gb']), state_shape)

        self.q = DeepQNetwork(self.learning_rate, state_shape, action_space.n)
        self.next_q = DeepQNetwork(self.learning_rate, state_shape, action_space.n)
        self.device = self.q.device

        self.learning_step = 0

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
        states, actions, rewards, next_states, dones = self._batch_as_tensors(batch_size)

        q_pred = self.q(states)
        q_pred = q_pred[np.arange(batch_size), actions.tolist()]

        q_next = self.next_q(next_states)
        q_next = q_next.max(dim=1)
        q_next = q_next[0]

        q_next[dones] = 0.0

        q_target = rewards + self.gamma * q_next

        loss = self.q.loss(q_pred, q_target).to(self.device)
        loss.backward()
        self.q.optimizer.step()

        self.learning_step += 1

    def _replace_target_network(self):
        """Syncs the target network with the main Q-network network."""
        if self.replace_every is None:
            return

        if self.learning_step % self.replace_every == 0:
            self.next_q.load_state_dict(self.q.state_dict())

    def _batch_as_tensors(self, batch_size):
        """Sample the batch then convert from NumPy arrays to Torch tensors."""
        states, actions, rewards, next_states, dones = self.replay_memory.sample(batch_size)

        t_states = torch.from_numpy(states).float().to(self.device)
        t_next_states = torch.from_numpy(next_states).float().to(self.device)

        t_actions = torch.from_numpy(actions).to(self.device)
        t_rewards = torch.from_numpy(rewards).float().to(self.device)
        t_dones = torch.from_numpy(dones).bool().to(self.device)

        return t_states, t_actions, t_rewards, t_next_states, t_dones

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_end)

    def store(self, observation, action, reward, next_observation, done):
        self.replay_memory.add_transition(observation, action, reward, next_observation, done)

    @property
    def checkpoint_data(self):
        return {
            'learning_step': self.learning_step,
            'q': self.q.state_dict(),
            'next_q': self.next_q.state_dict(),
            'optimizer': self.q.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }

    # def load_checkpoint(self, path):
    #     checkpoint = torch.load(path)
    #     self.q.load_state_dict(checkpoint['q'])
    #     self.next_q.load_state_dict(checkpoint['next_q'])
    #     self.q.optimizer.load_state_dict(checkpoint['optimizer'])
    #     self.epsilon = checkpoint['epsilon']
    #     self.learning_step = checkpoint['learning_step']
    #
    #     logging.info('Loaded checkpoint from %s.', path)
    #
    #     self.q.train()
    #     self.next_q.train()
