import torch
import torch.nn as nn
import torch.nn.functional as F

# https://medium.com/@mitchhuang777/image-recognition-with-cnns-improving-accuracy-and-efficiency-dd347b636e0c
class CNN_V1(nn.Module):
    def __init__(self):
        super(CNN_V1, self).__init__()
        # Parameters
        self.batch_size = 3500
        self.lr = 0.1
        self.momentun = 0.9

        # Hidden layers
        self.cl1 = nn.Conv2d(kernel_size=(5,5),in_channels=1, out_channels=5)
        self.mp1 = nn.MaxPool2d(kernel_size=(2,2))
        self.cl2 = nn.Conv2d(kernel_size=(5,5),in_channels=5, out_channels=10)
        self.mp2 = nn.MaxPool2d(kernel_size=(2,2))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(10*4*4, 32)
        self.fc2 = nn.Linear(32,10)
        
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentun)
        self.criterion = nn.MSELoss()
        
    def forward(self, x):
        x = F.relu(self.cl1(x))
        x = self.mp1(x)
        x = F.relu(self.cl2(x))
        x = self.mp2(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x