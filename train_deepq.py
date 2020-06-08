import argparse
import logging
import os
import sys

import numpy as np
import psutil
import torch
import yaml

from deepq.agent import Agent
from deepq.checkpoint_manager import CheckpointManager
from plot.plot import plot_scores, plot_ram
from wrappers.wrappers import make_env

logging.basicConfig(
    filename='./data/logs/training_log.log',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s: %(message)s',
)


def run_episode(env, agent, current_episode):
    score = 0
    done = False
    observation = env.reset()

    while not done:
        action = agent.act(observation)
        next_observation, reward, done, info = env.step(action)

        score += reward

        agent.store(observation, action, reward, next_observation, done)

        if current_episode >= 0:
            agent.learn()

        observation = next_observation

    agent.decay_epsilon()
    return score, agent.epsilon


def zip_everything():
    import shutil
    logger = logging.getLogger('zip_everything')
    shutil.make_archive('./results', 'gztar', '.', 'data', logger=logger)


def add_ram_usage(ram_values):
    process = psutil.Process(os.getpid())
    ram_values.append(process.memory_info().rss)  # in bytes


def shutdown_training(episode, scores, epsilons, plots_path, agent, checkpoint_manager, ram_values):
    checkpoint_manager.add(agent.checkpoint_data, scores[-1], episode)
    checkpoint_manager.log_data()

    plot_scores(scores, epsilons, plots_path)
    plot_ram(ram_values, plots_path)

    zip_everything()
    logging.info('Done.')


def get_parser():
    parser = argparse.ArgumentParser(description='Train a DQN agent')
    parser.add_argument('-c', '--conf', nargs='?', default=None, help='training configuration file', dest='filename',
                        required=True)

    return parser


def load_conf(path):
    with open(path, 'r') as stream:
        conf = yaml.safe_load(stream)

    conf_dir = os.path.dirname(path)
    to_absolute_paths(conf_dir, conf)

    return conf


def to_absolute_paths(root, d):
    for key in d.keys():
        if key == 'path' or key.endswith('_path'):
            d[key] = to_absolute(root, d[key])
            if not os.path.exists(d[key]):
                logging.error('Path \'%s\' for %s does not exist.', d[key], key)
        if type(d[key]) is dict:
            to_absolute_paths(root, d[key])


def to_absolute(root, path):
    return os.path.abspath(os.path.join(root, path))


def main(conf_file):
    conf = load_conf(conf_file)

    if torch.cuda.is_available():
        logging.info('GPU is available.')
    else:
        logging.info('GPU cannot be acquired.')

    env = make_env(conf['training']['gym_id'])
    env.seed(0)

    agent = Agent(conf['model'], action_space=env.action_space, state_shape=env.observation_space.shape)

    # Train
    batch_size = int(conf['model']['batch_size'])
    total_episodes = int(conf['training']['num_episodes'])

    scores = []
    epsilons = []
    ram_values = []
    episode = 0
    top_score = float('-inf')

    manager = CheckpointManager(conf['checkpoints'])

    # A negative episode index indicates that the agent does not learn,
    # but instead it just plays at random to gain some experience beforehand.
    for episode in range(-batch_size, total_episodes):
        try:
            score, epsilon = run_episode(env, agent, episode)
            top_score = max(top_score, score)

            # Performance metric for how well an agent does at a given point
            # is the mean performance over the last 100 steps.
            metric = np.mean(scores[-100:]).item() if len(scores) > 100 else np.mean(scores).item()
            manager.step(agent, metric, episode)

            scores.append(score)
            epsilons.append(epsilon)
            add_ram_usage(ram_values)

            logging.info(
                '[Episode %d] Score: %d, Top score: %.2f, Mean score (last 100): %.2f, Epsilon: %.3f',
                episode, score, top_score, metric, epsilon
            )

        except KeyboardInterrupt:
            logging.info('Gracefully shutting down...')
            shutdown_training(episode, scores, epsilons, conf['plots']['path'], agent, manager, ram_values)
            sys.exit()

    logging.info('Completed training.')
    shutdown_training(episode, scores, epsilons, conf['plots']['path'], agent, manager, ram_values)


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(args.filename)
