import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.model = nn.Sequential(
            nn.Conv2d(1, 6, 3),
            nn.ReLU(),
            self.pool,
            nn.Conv2d(6, 16, 3),
            nn.ReLU(),
            self.pool,
            nn.Flatten(),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Linear(200, 10),
        )
        

    def forward(self, x):
        return self.model(x)
