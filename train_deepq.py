import argparse
import logging
import os
import sys

import numpy as np
import torch
import yaml
import psutil

from deepq.agent import Agent
from deepq.checkpoint_manager import CheckpointManager
from deepq.replay_memory import ReplayMemory
from plot.plot import plot_scores, plot_ram
from wrappers.wrappers import make_env

logging.basicConfig(
    filename='./data/logs/training_log.log',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s: %(message)s',
)


def run_episode(env, agent, current_episode, batch_size):
    score = 0
    done = False
    observation = env.reset()

    while not done:
        action = agent.act(observation)
        next_observation, reward, done, info = env.step(action)

        score += reward

        agent.store(observation, action, reward, next_observation, done)

        if current_episode >= batch_size:
            agent.learn(batch_size)

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
    parser.add_argument('-c', '--conf', nargs='?', default=None, help='experiment configuration file', dest='filename',
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


def capacity_from_gb(gb: float, state_shape) -> int:
    rm = ReplayMemory(0, state_shape)
    array_size = state_shape[0] * state_shape[1] * state_shape[2]
    itemsize_sum = array_size * rm._states.itemsize + \
                   array_size * rm._next_states.itemsize + \
                   rm._actions.itemsize + rm._rewards.itemsize + rm._dones.itemsize
    return gb * 1.074e+9 // itemsize_sum


def main(conf_file):
    conf = load_conf(conf_file)

    if torch.cuda.is_available():
        logging.info('GPU is available.')
    else:
        logging.info('GPU cannot be acquired.')

    env = make_env(conf['training']['gym_id'])
    env.seed(0)

    model = conf['model']
    state_shape = env.observation_space.shape
    capacity = capacity_from_gb(model['memory_gb'], state_shape)

    agent = Agent(
        gamma=model['gamma'],
        epsilon=model['epsilon'],
        epsilon_end=model['epsilon_end'],
        epsilon_decay=model['epsilon_decay'],
        learning_rate=model['learning_rate'],
        memory_capacity=capacity,
        replace_every=model['replace_every'],
        action_space=env.action_space,
        state_shape=state_shape
    )

    # Train
    num_episodes = conf['training']['num_episodes']
    batch_size = model['batch_size']
    total_episodes = batch_size + num_episodes

    episode = 0
    scores = []
    epsilons = []
    top_score = float('-inf')
    ram_values = []

    manager = CheckpointManager(conf['checkpoints'])

    for episode in range(total_episodes):
        try:
            score, epsilon = run_episode(env, agent, episode, batch_size)

            mean_score = np.mean(scores) if scores else 0.0
            top_score = max(top_score, score)

            episode_index = episode - batch_size  # De-offset episode by batch_size.

            logging.info(
                '[Episode %d] Score: %d, Top score: %.2f, Avg. score: %.2f, Epsilon: %.3f',
                episode_index, score, top_score, mean_score, epsilon
            )

            manager.step(agent, score, episode_index)

            scores.append(score)
            epsilons.append(epsilon)
            add_ram_usage(ram_values)

        except KeyboardInterrupt:
            logging.info('Gracefully shutting down...')
            shutdown_training(episode, scores, epsilons, conf['plots']['path'], agent, manager, ram_values)
            sys.exit()

    logging.info('Completed training.')
    shutdown_training(episode, scores, epsilons, conf['plots']['path'], agent, manager, ram_values)


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(args.filename)
