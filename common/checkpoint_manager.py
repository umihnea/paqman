import bisect
import logging
import os

import torch
from dataclasses import dataclass

import datetime as datetime


@dataclass
class Checkpoint:
    time: datetime.datetime
    score: float
    episode: int
    filename: str


class CheckpointManager:
    def __init__(self, config):
        self.path = config["path"]
        self.max_size = int(config["max_size"])
        self.every = int(config["every"])

        self.checkpoints = []
        self.checkpoint_count = 0

        self.forced_checkpoints = []

    def step(self, agent, score: float, episode: int):
        if episode % self.every == 0:
            self.add(agent.checkpoint_data, score, episode)
            logging.info("Checkpoint taken at episode %d.", episode)

    def add(self, data, score: float, episode: int, commit=True):
        filename = "checkpoint_%d.pyt" % self.checkpoint_count
        checkpoint = Checkpoint(datetime.datetime.now(), score, episode, filename)

        scores = [c.score for c in self.checkpoints]
        index = bisect.bisect_left(scores, score)

        # Skip adding it if it's the smallest and we're out of space.
        if index == 0 and len(self.checkpoints) >= self.max_size:
            return

        self.checkpoint_count += 1
        self.checkpoints.insert(index, checkpoint)
        self._save_checkpoint(data, filename, commit)

        if len(self.checkpoints) > self.max_size:
            self._delete_checkpoint(self.checkpoints[0].filename, commit)
            self.checkpoints.pop(0)

    def force_add(self, data, score: float, episode: int, commit=True):
        """Add checkpoint files in a separate, uncontrolled list.

        Forced checkpoints are not limited and are not compared against
        previous entries.
        """
        filename = "forced_save_%d.pyt" % len(self.forced_checkpoints)
        checkpoint = Checkpoint(datetime.datetime.now(), score, episode, filename)
        self._save_checkpoint(data, filename, commit)
        self.forced_checkpoints.append(checkpoint)

    def _save_checkpoint(self, data, filename, commit):
        if not commit:
            return

        path = os.path.join(self.path, filename)
        torch.save(data, path)

    def _delete_checkpoint(self, filename, commit):
        if not commit:
            return

        path = os.path.join(self.path, filename)
        os.remove(path)

    def log_data(self):
        path = os.path.join(self.path, "checkpoint_data.txt")
        table = []
        headers = ["filename", "episode", "score", "timestamp"]
        with open(path, "w") as f:
            for point in self.checkpoints + self.forced_checkpoints:
                timestamp = point.time.strftime("%d-%m-%Y %H:%M:%S")
                table.append([point.filename, point.episode, point.score, timestamp])

            from tabulate import tabulate

            f.write(tabulate(table, headers=headers))
