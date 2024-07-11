import torch
import torch.nn as nn
import torch.nn.functional as F

class RoundLayer(nn.Module):
    def __init__(self):
        super(RoundLayer, self).__init__()

    def forward(self, x):
        return torch.round(x)

class MLP_V1(nn.Module):
    def __init__(self, input_size, lr):
        super(MLP_V1, self).__init__()
        self.input_size = input_size * input_size
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.input_size, self.input_size//2)
        self.fc2 = nn.Linear(self.input_size//2, self.input_size//4)
        self.fc3 = nn.Linear(self.input_size//4, 1)  # Output a single scalar value
        self.round = RoundLayer()
        
        # Initialize the optimizer and criterion
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.round(x)
        return x
