import torch
import torch.nn as nn
import torch.nn.functional as F

class MushroomClassifier(nn.Module):
    def __init__(self):
        super(MushroomClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # Dynamically calculate flatten size
        self._to_linear = None
        self.calculate_linear_input()

        self.fc1 = nn.Linear(self._to_linear, 64)
        self.fc2 = nn.Linear(64, 2)

    def calculate_linear_input(self):
        # Pass a dummy image through conv layers to find the flatten size
        x = torch.randn(1, 3, 64, 64)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        self._to_linear = x.view(1, -1).shape[1]

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
