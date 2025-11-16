import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 200)
        self.fc2 = nn.Linear(200, 10)  

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
