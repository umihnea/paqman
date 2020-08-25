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

    def learn(self, batch_size: int = 32):
        self._replace_target_network()
        self._learn_using(self.q, self.next_q, batch_size)
        self.learning_step += 1

    def _learn_using(
        self, q_one: DeepQNetwork, q_two: DeepQNetwork, batch_size: int = 32
    ):
        """Calling this function requires specifying which network will serve
        as the action-picker and which one is the evaluator.

        :param q_one: this is the network which learns and which picks the actions
        :param q_two: this is the network which estimates the values of the picked actions
        :param batch_size: the size of a sampled batch
        """
        q_one.optimizer.zero_grad()

        states, actions, rewards, next_states, dones = self.replay_memory.sample(
            batch_size
        ).as_tensors(self.device)

        q_pred = q_one(states)
        q_pred = q_pred[np.arange(batch_size), actions.tolist()]

        # Double Q-learning requires us to chose the actions separately
        # chosen_actions = [argmax Q1(s', a)]
        chosen_actions = q_one(next_states).argmax(dim=1)

        # predicted_values = [Q2(s', [chosen_actions])]
        predicted_values = q_two(next_states)
        predicted_values = predicted_values[torch.arange(batch_size), chosen_actions]
        predicted_values[dones] = 0.0

        q_target = rewards + self.gamma * predicted_values

        loss = q_one.loss(q_pred, q_target).to(self.device)
        loss.backward()
        q_one.optimizer.step()
