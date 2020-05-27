import argparse

from gym import wrappers, logger

from wrappers.wrappers import make_env


class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space
        logger.debug('action: shape=%s, dtype=%s', action_space.shape, action_space.dtype)

    def act(self, observation, reward, done):
        logger.debug('observation: shape=%s, dtype=%s', observation.shape, observation.dtype)
        return self.action_space.sample()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env', nargs='?', default='MsPacman-v4', help='Select the environment to run')
    args = parser.parse_args()

    logger.set_level(logger.DEBUG)  # INFO

    env = make_env(args.env)

    # Connect monitor
    output_directory = './data/random-results'
    env = wrappers.Monitor(env, directory=output_directory, force=True)
    env.seed(0)

    # Initialize agent
    agent = RandomAgent(env.action_space)

    episode_count = 100
    reward = 0
    done = False

    # for i in range(episode_count):
    ob = env.reset()
    while True:
        action = agent.act(ob, reward, done)
        ob, reward, done, _ = env.step(action)
        if done:
            break

    print(ob)

    env.close()
