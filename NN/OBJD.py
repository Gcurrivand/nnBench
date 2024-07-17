import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from CNN import LeNet5

# OBJD will run on image of size 560 x 560 with LeNet5 as CNN
class OBJD_LeNet5(nn.Module):
    def __init__(self):
        super(OBJD_LeNet5, self).__init__()
        self.model = LeNet5()
        self.lr = 0.1
        
        self.optimizer = torch.optim.Adagrad(self.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
    def forward(self, x):
        # Load the model weights
        model_path = os.path.join("../Models/CNN/lenet5", "best_lenet5_weights.pth")
        self.model.load_state_dict(torch.load(model_path))
        # Set the model to evaluation mode
        self.model.eval()
        
        return x
    
    def segmentation():
