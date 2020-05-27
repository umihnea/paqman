import argparse

import numpy as np
from gym import wrappers, logger

from deepq.agent import Agent
from plot.plot import plot_to_file
from wrappers.wrappers import make_env

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--episodes', nargs='?', default='100', help='Number of games to play')
    parser.add_argument('--env', nargs='?', default='MsPacman-v4', help='The environment to run')
    args = parser.parse_args()

    logger.set_level(logger.INFO)

    env = make_env(args.env)

    # Connect monitor
    output_directory = './data/deepq-results'
    env = wrappers.Monitor(env, directory=output_directory, force=True)
    env.seed(0)

    agent = Agent(
        action_space=env.action_space,
        state_shape=env.observation_space.shape,
    )

    # Train
    total_episodes = int(args.episodes) or 100

    scores = []
    epsilons = []
    batch_size = 11  # 32

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

            # Penalize for losing a life
            # current_lives = info['ale.lives']
            # if current_lives < lives:
            #     reward = -100

            agent.store(observation, action, reward, next_observation, done)

            if episode >= batch_size:
                agent.learn(batch_size)

            observation = next_observation

        scores.append(score)

        top_score = np.amax(scores)
        mean_score = np.mean(scores)
        logger.info(
            '[Episode %d] Score: %d, Top score: %.2f, Avg. score: %.2f',
            episode, score, top_score, mean_score
        )

        epsilons.append(agent.epsilon)

    plot_to_file(scores, epsilons, len(scores), './data/deepq-results')
