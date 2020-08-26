import numpy as np
import torch
from gym.spaces import Discrete

from deepq.agent import Agent
from deepq.network import DeepQNetwork


class DoubleDQNAgent(Agent):
    def __init__(
        self, model_parameters, action_space=Discrete(4), state_shape=(4, 84, 84)
    ):
        # Keep the same configuration as the DQN agent.
        super().__init__(model_parameters, action_space, state_shape)

    def _compute_double_learning_target(
        self,
        q_one: DeepQNetwork,
        q_two: DeepQNetwork,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        batch_size: int,
    ):
        # Double Q-learning requires us to chose the actions separately.
        # chosen_actions = [argmax Q1(s', a)]
        chosen_actions = q_one(next_states).argmax(dim=1)

        # predicted_values = [Q2(s', [chosen_actions])]
        predicted_values = q_two(next_states)
        predicted_values = predicted_values[torch.arange(batch_size), chosen_actions]
        predicted_values[dones] = 0.0

        return rewards + self.gamma * predicted_values

    def learn(self, batch_size: int = 32):
        self._replace_target_network()
        self.q.optimizer.zero_grad()

        states, actions, rewards, next_states, dones = self.replay_memory.sample(
            batch_size
        ).as_tensors(self.device)

        q_pred = self.q(states)
        q_pred = q_pred[np.arange(batch_size), actions.tolist()]

        q_target = self._compute_double_learning_target(
            self.q, self.next_q, rewards, next_states, dones, batch_size
        )

        loss = self.q.loss(q_pred, q_target).to(self.device)
        loss.backward()
        self.q.optimizer.step()

        self.learning_step += 1
