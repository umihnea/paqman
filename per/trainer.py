import logging
import os
import pickle
import sys
from datetime import datetime
from typing import Tuple

import numpy as np
import psutil
import torch

from deepq.checkpoint_manager import CheckpointManager
from deepq.conf_loader import ConfLoader
from per.agent import PERAgent
from wrappers.wrappers import make_env


class Trainer:
    def __init__(self, path_to_config):
        self.conf = ConfLoader(path_to_config).load()
        self._log_cuda_status()

        self.env = make_env(self.conf["training"]["gym_id"])
        self.env.seed(0)

        self.agent = PERAgent(
            self.conf["model"],
            action_space=self.env.action_space,
            state_shape=self.env.observation_space.shape,
        )

        self.manager = CheckpointManager(self.conf["checkpoints"])

        self.batch_size = int(self.conf["model"]["batch_size"])
        self.total_episodes = int(self.conf["training"]["num_episodes"])

        self.process = psutil.Process(os.getpid())

        self.scores = []
        self.epsilons = []
        self.memory_usage = []
        self.top_score = float("-inf")

    @staticmethod
    def _log_cuda_status():
        if torch.cuda.is_available():
            logging.info("GPU is available.")
        else:
            logging.info("GPU cannot be acquired.")

    def train(self):
        """This is the main training loop.

        A negative episode index indicates that the agent does not
        learn, but instead it just plays at random to gain some
        experience beforehand.
        """
        for episode in range(-self.batch_size, self.total_episodes):
            try:
                score, epsilon = self.run_episode(episode)
                self.top_score = max(self.top_score, score)

                # Performance metric for how well an agent does at a given point
                # is the mean performance over the last 100 steps.
                metric = self._compute_metric()
                self.manager.step(self.agent, metric, episode)

                self.scores.append(score)
                self.epsilons.append(epsilon)
                self.memory_usage.append(self.process.memory_info().rss)

                logging.info(
                    "[Episode %d] Score: %.1f, Top score: %.2f, Mean score (last 100): %.2f, Epsilon: %.3f",
                    episode,
                    score,
                    self.top_score,
                    metric,
                    epsilon,
                )

            except KeyboardInterrupt:
                logging.info("Gracefully shutting down...")
                self.shutdown(episode)
                sys.exit()

        logging.info("Completed training.")
        self.shutdown(self.total_episodes)

    def _compute_metric(self) -> float:
        if len(self.scores) == 0:
            return 0.0
        return (
            np.mean(self.scores[-100:]).item()
            if len(self.scores) > 100
            else np.mean(self.scores).item()
        )

    def run_episode(self, episode) -> Tuple[float, float]:
        score = 0.0
        done = False
        observation = self.env.reset()

        while not done:
            action = self.agent.act(observation)
            next_observation, reward, done, info = self.env.step(action)

            score += reward

            self.agent.store(observation, action, reward, next_observation, done)

            if episode >= 0:
                self.agent.learn()

            observation = next_observation

        self.agent.decay_epsilon()
        return score, self.agent.epsilon

    def shutdown(self, episode):
        self.manager.force_add(
            self.agent.checkpoint_data, self._compute_metric(), episode
        )
        self.manager.log_data()

        # Pickle lists for analysis by outside tools.
        self._pickle_data(self.scores, "scores")
        self._pickle_data(self.epsilons, "epsilons")
        self._pickle_data(self.memory_usage, "memory_usage")

        # Compress data into a .tar.gz archive.
        import shutil

        shutil.make_archive(
            "data/results", "gztar", ".", "data", logger=logging.getLogger("zip_data")
        )

        logging.info("Done.")
        logging.shutdown()

    def _pickle_data(self, data, base_name):
        """Pickle data to a file named based on the nickname of the current
        training process and a base name which describes what the file
        contains."""
        filename = "{base_name}_{name}_{date}.pkl".format(
            base_name=base_name,
            name=self.conf["training"]["name"],
            date=datetime.now().strftime("%d%b_%H-%M-%S"),
        )
        path = os.path.join(self.conf["logs"]["path"], filename)

        with open(path, "wb") as file:
            pickle.dump(data, file)
