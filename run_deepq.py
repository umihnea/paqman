import argparse
import os

import numpy as np
import torch
from gym import wrappers, logger

from deepq.agent import Agent
from plot.plot import plot_to_file
from wrappers.wrappers import make_env

DATA_ROOT = './data/deepq'
PLOTS_DIR = '%s/plots' % DATA_ROOT
CHECKPOINTS_DIR = '%s/checkpoints' % DATA_ROOT
MONITOR_DIR = '%s/monitor' % DATA_ROOT

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--episodes', nargs='?', default=100, help='Number of episodes to play for')
    parser.add_argument('--env', nargs='?', default='MsPacman-v4', help='The environment to run')
    parser.add_argument('--batch', nargs='?', default=32, help='Batch size hyperparameter')
    parser.add_argument('--checkpoint', nargs='?', default=None, help='Checkpoint file to reload training from')
    args = parser.parse_args()

    logger.set_level(logger.INFO)

    env = make_env(args.env)

    # Connect monitor
    env = wrappers.Monitor(env, directory=MONITOR_DIR, force=True)
    env.seed(0)

    agent = Agent(
        action_space=env.action_space,
        state_shape=env.observation_space.shape,
    )

    if args.checkpoint is not None:
        path_to_checkpoint = '/'.join([CHECKPOINTS_DIR, args.checkpoint])
        if not os.path.isfile(path_to_checkpoint):
            logger.error('Checkpoint file \'%s\' not found.', path_to_checkpoint)
        else:
            agent.load_checkpoint(path_to_checkpoint)

    # Check CUDA
    if torch.cuda.is_available():
        logger.info('GPU is available!')
    else:
        logger.info('GPU cannot be acquired.')

    # Train
    num_episodes = int(args.episodes)
    batch_size = int(args.batch)
    total_episodes = batch_size + num_episodes

    scores = []
    epsilons = []
    top_score = 0.0

    for episode in range(total_episodes):
        score = 0
        step_count = 0
        done = False
        observation = env.reset()

        lives = 0

        while not done:
            action = agent.act(observation)
            next_observation, reward, done, info = env.step(action)

            score += reward

            agent.store(observation, action, reward, next_observation, done)

            if episode >= batch_size:
                agent.learn(batch_size)

            observation = next_observation

        scores.append(score)

        # Update maximum and save checkpoint
        current_max = np.amax(scores)
        if top_score < current_max:
            if episode > batch_size:
                agent.save_checkpoint(CHECKPOINTS_DIR)

            top_score = current_max

        mean_score = np.mean(scores)
        logger.info(
            '[Episode %d] Score: %d, Top score: %.2f, Avg. score: %.2f',
            episode - batch_size, score, top_score, mean_score
        )

        epsilons.append(agent.epsilon)

    plot_to_file(scores, epsilons, len(scores), PLOTS_DIR)
