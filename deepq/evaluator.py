import logging

from deepq.agent import Agent
from deepq.conf_loader import ConfLoader
from wrappers.wrappers import make_env


class Evaluator:
    """Contains the evaluation logic for trained agents."""

    def __init__(self, path_to_config, path_to_checkpoint):
        conf = ConfLoader(path_to_config).load()
        self.plots_path = conf["plots"]["path"]
        self.num_episodes = int(conf["evaluation"]["num_episodes"])

        # Wrap environment in Monitor wrapper
        env = make_env(conf["training"]["gym_id"], monitor_path=conf["monitor"]["path"])
        env.seed(0)
        self.env = env

        self.agent = Agent.from_checkpoint(path_to_checkpoint, conf["model"], env)
        self.agent.toggle_eval()

        self.scores = []

    def evaluate(self):
        """Runs the main loop then parses the data."""
        for episode in range(self.num_episodes):
            score = self._run_episode()
            self.scores.append(score)

            logging.info("[Episode %d] Score: %.2f", episode, score)

        from plot.plot import plot_evaluation

        plot_evaluation(self.scores, self.plots_path)

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
