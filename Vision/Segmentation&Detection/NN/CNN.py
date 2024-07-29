import torch
import torch.nn as nn
import torch.nn.functional as F
import os
    
# https://www.datasciencecentral.com/lenet-5-a-classic-cnn-architecture/
# As image is 28x28x1 the first convolution is used to to it into 28x28x6 to follow LeNet5 architecture, using this formula new shape [(W−K+2P)/S]+1. 
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # Parameters
        self.batch_size = 3500
        self.lr = 0.1
        self.momentun = 0.9
        self.name = "lenet5"

        # Hidden layers
        self.cl1 = nn.Conv2d(kernel_size=(3,3), padding=1,in_channels=1, out_channels=6)
        self.avgp = nn.AvgPool2d(kernel_size=(2,2), stride=2)
        self.cl2 = nn.Conv2d(kernel_size=(5,5),in_channels=6, out_channels=16)
        self.cl3 = nn.Conv2d(kernel_size=(5,5),in_channels=16, out_channels=120)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)
        
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentun)
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        x = F.tanh(self.cl1(x))
        x = F.tanh(self.avgp(x))
        x = F.tanh(self.cl2(x))
        x = F.tanh(self.avgp(x))
        x = F.tanh(self.cl3(x))
        x = self.flatten(x)
        x = F.tanh(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x

def cnn_run_single_inference(model, image):
    image = load_and_prepare_image(image)
    model_path = os.path.join(os.path.join(os.path.dirname(__file__),f"../Models/CNN/{model.name}/best_{model.name}_weights.pth"))
    model.load_state_dict(torch.load(model_path))
    model.eval() 
    image = image.unsqueeze(0)
    output = model(image)
    prediction = torch.argmax(output)
    print(prediction)
    return prediction

def cnn_run_multiple_inference(model, images):
    output = model(images)
    predictions = torch.argmax(output, dim=1)
    return predictions
    