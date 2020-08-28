from gym.spaces import Discrete


class RandomAgent:
    """This agent has been built for evaluation only."""

    def __init__(
        self, model_parameters, action_space=Discrete(4), state_shape=(4, 84, 84)
    ):
        self.action_space = action_space

    def act(self, _):
        """Act randomly by sampling the action space.

        :return action: an action in the action space
        """
        return self.action_space.sample()
