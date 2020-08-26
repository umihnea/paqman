import numpy as np
import torch
from gym.spaces import Discrete

from deepq.agent import Agent


class DoubleDQNAgent(Agent):
    def __init__(
        self, model_parameters, action_space=Discrete(4), state_shape=(4, 84, 84)
    ):
        # Keep the same configuration as the DQN agent.
        super().__init__(model_parameters, action_space, state_shape)

    def learn(self, batch_size: int = 32):
        self._replace_target_network()

        # q_one is the action picker and q_two is the action evaluator.
        # The action picker is also the network that is updated.
        q_one = self.q
        q_two = self.next_q

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

        self.learning_step += 1
