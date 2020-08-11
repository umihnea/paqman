import numpy as np
import torch
from gym.spaces import Discrete

from deepq.agent import Agent
from replay_memory.prioritized_replay_memory import PrioritizedReplayMemory


class PERAgent(Agent):
    def __init__(
        self, model_parameters, action_space=Discrete(4), state_shape=(4, 84, 84)
    ):
        super().__init__(model_parameters, action_space, state_shape)
        per_alpha = float(model_parameters["per_alpha"])
        per_epsilon = float(model_parameters["per_epsilon"])
        self.replay_memory = PrioritizedReplayMemory(
            float(model_parameters["memory_gb"]), state_shape, per_alpha, per_epsilon
        )

    def learn(self, batch_size=32):
        self.q.optimizer.zero_grad()
        self._replace_target_network()

        batch, indices = self.replay_memory.sample(batch_size)
        states, actions, rewards, next_states, dones = batch.as_tensors(self.device)

        q_pred = self.q(states)
        q_pred = q_pred[np.arange(batch_size), actions.tolist()]

        q_next = self.next_q(next_states)
        q_target = self._compute_q_target(q_next, rewards, dones)

        error = self._compute_error(q_target, q_pred)
        self.replay_memory.batch_update(indices, error)

        loss = self.q.loss(q_pred, q_target).to(self.device)
        loss.backward()
        self.q.optimizer.step()

        self.learning_step += 1

    @staticmethod
    def _compute_error(q_target: torch.tensor, q_pred: torch.tensor) -> np.array:
        """In classic DQN we can get by without computing the error manually.
        Torch's loss functions take care of that. In prioritized replay, however,
        we need the errors to compute priorities. This function additionally has
        to copy the tensors back to main memory and convert them to numpy arrays."""
        q_target_numpy = q_target.detach().cpu().numpy()
        q_pred_numpy = q_pred.detach().cpu().numpy()
        return np.abs(q_target_numpy - q_pred_numpy)
