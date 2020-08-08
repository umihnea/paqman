import numpy as np
import torch
from gym.spaces import Discrete

from deepq.network import DeepQNetwork
from replay_memory.prioritized_replay_memory import PrioritizedReplayMemory


class PERAgent:
    def __init__(
        self, model_parameters, action_space=Discrete(4), state_shape=(4, 84, 84)
    ):
        self.learning_rate = float(model_parameters["learning_rate"])
        self.gamma = float(model_parameters["gamma"])

        self.epsilon = float(model_parameters["epsilon"])
        self.epsilon_end = float(model_parameters["epsilon_end"])
        self.epsilon_decay = float(model_parameters["epsilon_decay"])
        self.batch_size = int(model_parameters["batch_size"]) or 32
        self.replace_every = int(model_parameters["replace_every"])

        self.action_space = action_space

        # PER
        per_alpha = float(model_parameters["per_alpha"])
        per_epsilon = float(model_parameters["per_epsilon"])
        self.replay_memory = PrioritizedReplayMemory(
            float(model_parameters["memory_gb"]), state_shape, per_alpha, per_epsilon
        )

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

        batch, indices = self.replay_memory.sample(batch_size)
        states, actions, rewards, next_states, dones = batch.as_tensors(self.device)

        q_pred = self.q(states)
        q_pred = q_pred[np.arange(batch_size), actions.tolist()]

        q_next = self.next_q(next_states)
        q_next = q_next.max(dim=1)
        q_next = q_next[0]

        q_next[dones] = 0.0

        q_target = rewards + self.gamma * q_next

        # PER
        qt = q_target.detach().cpu().numpy()
        qp = q_pred.detach().cpu().numpy()
        error = np.abs(qt - qp)
        self.replay_memory.batch_update(indices, error)

        loss = self.q.loss(q_pred, q_target).to(self.device)
        loss.backward()
        self.q.optimizer.step()

        self.learning_step += 1

    def _replace_target_network(self):
        """Sync the target network to the main network.
        """
        if self.replace_every is None:
            return

        if self.learning_step % self.replace_every == 0:
            self.next_q.load_state_dict(self.q.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_end)

    def store(self, observation, action, reward, next_observation, done):
        self.replay_memory.add_transition(
            observation, action, reward, next_observation, done
        )

    @property
    def checkpoint_data(self):
        return {
            "learning_step": self.learning_step,
            "q": self.q.state_dict(),
            "next_q": self.next_q.state_dict(),
            "optimizer": self.q.optimizer.state_dict(),
            "epsilon": self.epsilon,
        }

    @classmethod
    def from_checkpoint(cls, path, conf, env):
        """Load from PyTorch checkpoint file.
        See custom format of file in checkpoint_data.
        """
        conf["memory_gb"] = 0  # Don't allocate memory for replay buffer

        agent = cls(
            conf, action_space=env.action_space, state_shape=env.observation_space.shape
        )

        # agent.device already contains the correct device,
        # depending on what is available on the machine.
        checkpoint = torch.load(path, map_location=agent.device)

        agent.q.load_state_dict(checkpoint["q"])
        agent.next_q.load_state_dict(checkpoint["next_q"])
        agent.q.optimizer.load_state_dict(checkpoint["optimizer"])
        agent.epsilon = float(checkpoint["epsilon"])
        agent.learning_step = int(checkpoint["learning_step"])

        return agent

    def toggle_eval(self):
        self.q.eval()
        self.next_q.eval()

    def toggle_training(self):
        self.q.train()
        self.next_q.train()
