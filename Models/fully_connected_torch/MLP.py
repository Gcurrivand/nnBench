import torch
import torch.nn as nn
import torch.nn.functional as F

image_height = 30
image_width = 30
input_size = image_height * image_width

class MLP_V1(nn.Module):
    def __init__(self):
        super(MLP_V1, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(self.input_size, self.input_size // 9)
        self.fc2 = nn.Linear(self.input_size // 9, 10)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x