from common.trainer import Trainer
from per.agent import PERAgent


class PERTrainer(Trainer):
    def __init__(self, path_to_config):
        super().__init__(path_to_config)

    def _create_agent(self):
        return PERAgent(
            self.conf["model"],
            action_space=self.env.action_space,
            state_shape=self.env.observation_space.shape,
        )
