import logging
import os
import pickle

from common.trainer import Trainer
from deepq.agent import Agent
from common.config_loader import ConfigLoader
from randomagent.agent import RandomAgent
from wrappers.wrappers import make_env


class Evaluator:
    """Contains the evaluation logic for trained agents."""

    def __init__(self, path_to_config, path_to_checkpoint, random=False):
        conf = ConfigLoader(path_to_config).load()
        self.num_episodes = int(conf["evaluation"]["num_episodes"])
        self.label = conf["training"]["name"]
        self.logs_path = conf["logs"]["path"]

        # Wrap environment in Monitor wrapper
        env = make_env(conf["training"]["gym_id"], monitor_path=conf["monitor"]["path"])
        env.seed(0)
        self.env = env

        if not random:
            self.agent = Agent.from_checkpoint(path_to_checkpoint, conf["model"], env)
            self.agent.toggle_eval()
        else:
            self.agent = RandomAgent(
                conf["model"],
                action_space=self.env.action_space,
                state_shape=self.env.observation_space.shape,
            )

        self.scores = []

    def evaluate(self):
        """Runs the main loop then parses the data."""
        for episode in range(self.num_episodes):
            score = self._run_episode()
            self.scores.append(score)

            logging.info("[Episode %d] Score: %.2f", episode, score)

        self._evaluation_pickle_data(self.scores, "scores")

    def _run_episode(self) -> float:
        score = 0.0
        done = False
        observation = self.env.reset()

        while not done:
            action = self.agent.act(observation)
            next_observation, reward, done, info = self.env.step(action)
            score += reward
            observation = next_observation

        return score

    def _evaluation_pickle_data(self, data, base_name):
        """See Trainer._pickle_data."""
        filename = "evaluation_{base_name}_{name}_{date}.pkl".format(
            base_name=base_name, name=self.label, date=Trainer._now(),
        )
        path = os.path.join(self.logs_path, filename)

        with open(path, "wb") as file:
            pickle.dump(data, file)
