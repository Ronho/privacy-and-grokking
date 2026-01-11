import torch
import torch.nn as nn
import torch.nn.functional as F
from math import floor


class CNN(nn.Module):
    def __init__(self, input_dim: torch.Size, num_classes: int):
        super().__init__()
        c, h, w = input_dim
        self.conv1 = nn.Conv2d(c, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)

        def _conv_out_dim(in_dim, kernel_size=3, stride=1, padding=0):
            return floor((in_dim + 2 * padding - (kernel_size - 1) - 1) / stride + 1)

        def _pool_out_dim(in_dim, kernel_size=2, stride=2, padding=0):
            return floor((in_dim + 2 * padding - (kernel_size - 1) - 1) / stride + 1)

        h = _conv_out_dim(h)
        w = _conv_out_dim(w)
        h = _pool_out_dim(h)
        w = _pool_out_dim(w)
        h = _conv_out_dim(h)
        w = _conv_out_dim(w)
        h = _pool_out_dim(h)
        w = _pool_out_dim(w)

        conv_output_size = 16 * h * w

        self.fc1 = nn.Linear(conv_output_size, 200)
        self.fc2 = nn.Linear(200, num_classes)

    def forward(self, input):
        y = self.conv1(input)
        y = F.relu(y)
        y = F.max_pool2d(y, 2, 2)
        y = self.conv2(y)
        y = F.relu(y)
        y = F.max_pool2d(y, 2, 2)
        y = torch.flatten(y, 1)
        y = self.fc1(y)
        y = F.relu(y)
        y = self.fc2(y)
        return y

    @property
    def last_layer(self):
        return self.fc2
