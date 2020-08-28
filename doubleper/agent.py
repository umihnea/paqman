from gym.spaces import Discrete
import numpy as np

from doubledqn.agent import DoubleDQNAgent
from per.agent import PERAgent
from replay_memory.prioritized_replay_memory import PrioritizedReplayMemory


class DoublePERAgent(DoubleDQNAgent):
    def __init__(
        self, model_parameters, action_space=Discrete(4), state_shape=(4, 84, 84)
    ):
        super().__init__(model_parameters, action_space, state_shape)
        per_alpha = float(model_parameters["per_alpha"])
        per_epsilon = float(model_parameters["per_epsilon"])
        self.replay_memory = PrioritizedReplayMemory(
            float(model_parameters["memory_gb"]), state_shape, per_alpha, per_epsilon
        )

    def learn(self, batch_size: int = 32):
        """This procedure is the same as doubledqn.DoubleDQNAgent.learn with
        the exception that it handles a prioritized replay memory which
        requires keeping track of indices and updating their priorities."""
        self._replace_target_network()
        self.q.optimizer.zero_grad()

        batch, indices = self.replay_memory.sample(batch_size)
        states, actions, rewards, next_states, dones = batch.as_tensors(self.device)

        q_pred = self.q(states)
        q_pred = q_pred[np.arange(batch_size), actions.tolist()]

        q_target = self._compute_double_learning_target(
            self.q, self.next_q, rewards, next_states, dones, batch_size
        )

        error = PERAgent._compute_error(
            q_target, q_pred
        )  # fixme: check if this is computed correctly
        self.replay_memory.batch_update(indices, error)

        loss = self.q.loss(q_pred, q_target).to(self.device)
        loss.backward()
        self.q.optimizer.step()

        self.learning_step += 1
