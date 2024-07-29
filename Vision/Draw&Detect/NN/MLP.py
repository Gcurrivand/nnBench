import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP_V1(nn.Module):
    def __init__(self):
        super(MLP_V1, self).__init__()
        self.batch_size = 3500
        self.input_size = 28*28
        self.lr = 0.1
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.input_size, 784)
        self.fc2 = nn.Linear(784, 450)
        self.fc3 = nn.Linear(450, 1)
        
        self.optimizer = torch.optim.Adagrad(self.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x
    
class MLP_V2(nn.Module):
    def __init__(self):
        super(MLP_V2, self).__init__()
        self.batch_size = 3500
        self.input_size = 28*28
        self.lr = 0.05
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.input_size, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 10)
        
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x