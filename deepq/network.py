import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import RMSprop


class DeepQNetwork(nn.Module):
    def __init__(self, learning_rate, input_shape, output_size):
        super(DeepQNetwork, self).__init__()

        self.conv1 = nn.Conv2d(input_shape[0], 32, 8, stride=4)  # input_shape[0] is stack depth
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1)

        linear_in = self._calculate_conv_output_dims(input_shape)

        self.linear1 = nn.Linear(linear_in, 512)
        self.linear2 = nn.Linear(512, output_size)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.loss = nn.MSELoss()
        self.optimizer = RMSprop(self.parameters(), lr=learning_rate)

    def _calculate_conv_output_dims(self, input_dims):
        state = torch.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, observation):
        """Compute the forward pass of the Q-network.
        :param observation: nd-array representing the stack of emulator frames
        :return: q_values: Q-value of each action
        """
        x = F.relu(self.conv1(observation))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)

        x = F.relu(self.linear1(x))
        return self.linear2(x)
