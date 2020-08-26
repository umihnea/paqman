from common.trainer import Trainer
from .agent import DoublePERAgent


class DoublePERTrainer(Trainer):
    def __init__(self, path_to_config):
        super().__init__(path_to_config)

    def _create_agent(self):
        return DoublePERAgent(
            self.conf["model"],
            action_space=self.env.action_space,
            state_shape=self.env.observation_space.shape,
        )
