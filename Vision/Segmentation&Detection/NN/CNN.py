import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
package_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, package_path)
from DatasetReader import CustomDataset, PathImageToTensor, PILImageToTensor
from torch.utils.data import DataLoader

cfp = os.path.join(os.path.dirname(__file__))
    
# https://www.datasciencecentral.com/lenet-5-a-classic-cnn-architecture/
# As image is 1x28x28 the first convolution is used to turn it into 6x28x28 to follow LeNet5 architecture, using this formula new shape [(W−K+2P)/S]+1. 
class LeNet5(nn.Module):
    def __init__(self, lr= 0.1, momentum= 0.9):
        super(LeNet5, self).__init__()
        self.name = "lenet5"
        self.lr = lr
        self.momentum = momentum

        self.cl1 = nn.Conv2d(kernel_size=(5,5),in_channels=1, out_channels=6)
        self.avgp = nn.AvgPool2d(kernel_size=(2,2), stride=2)
        self.cl2 = nn.Conv2d(kernel_size=(5,5),in_channels=6, out_channels=16)
        self.cl3 = nn.Conv2d(kernel_size=(5,5),in_channels=16, out_channels=120)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(120, 100)
        self.fc2 = nn.Linear(100, 81)
        
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
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
    
class LeNet5ReluMax(nn.Module):
    def __init__(self, lr= 0.1, momentum= 0.9):
        super(LeNet5ReluMax, self).__init__()
        self.name = "lenet5"
        self.lr = lr
        self.momentum = momentum

        self.cl1 = nn.Conv2d(kernel_size=(5,5),in_channels=1, out_channels=6)
        self.maxp = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.cl2 = nn.Conv2d(kernel_size=(5,5),in_channels=6, out_channels=16)
        self.cl3 = nn.Conv2d(kernel_size=(5,5),in_channels=16, out_channels=120)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(120, 100)
        self.fc2 = nn.Linear(100, 81)
        
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        x = F.relu(self.cl1(x))
        x = F.relu(self.maxp(x))
        x = F.relu(self.cl2(x))
        x = F.relu(self.maxp(x))
        x = F.relu(self.cl3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x

def compute_inference(model, tensor_input, weights_path=None):
    if not weights_path:
        weights_path = f"./Weights/best_{model.name}_weights.pth"
    model.load_state_dict(torch.load(os.path.join(cfp,weights_path)))
    model.eval() 
    output = model(tensor_input)
    print(output)
    prediction = torch.argmax(output)
    print(prediction)
    return prediction

def cnn_run_single_inference_image_path(model, img_path, weights_path=None):
    if not weights_path:
        weights_path = f"./Weights/best_{model.name}_weights.pth"
    tensor_input = PathImageToTensor(img_path).unsqueeze(0)
    return compute_inference(model, tensor_input, weights_path)

def cnn_run_single_inference_image_pil(model, img, weights_path=None):
    if not weights_path:
        weights_path = f"./Weights/best_{model.name}_weights.pth"
    tensor_input = PILImageToTensor(img).unsqueeze(0)
    return compute_inference(model, tensor_input, weights_path)

def cnn_run_multiple_inference(model, images):
    output = model(images)
    predictions = torch.argmax(output, dim=1)
    return predictions

def train(model, nb_epochs, batch_size, weights_path=None, data_path="../Dataset/Data/Train"):
    if not weights_path:
        weights_path = f"./Weights/best_{model.name}_weights.pth"
    print("Data are processed")
    train_data = CustomDataset(os.path.join(cfp,data_path,"data.csv"), os.path.join(cfp,data_path))
    dataloader = DataLoader(train_data, batch_size, shuffle=True, num_workers=0)
    print("Data processed")
    best_loss = float('inf')
    best_model_path = os.path.join(os.path.dirname(__file__), weights_path)
    for epoch in range(nb_epochs):
        epoch_loss = 0
        for batch_idx,(data, labels) in enumerate(dataloader):
            outputs = model(data)
            loss = model.criterion(outputs, labels)
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{nb_epochs}], Loss: {avg_loss:.4f}')
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with loss: {best_loss:.4f}")
    print("Training complete.")
    print(f"Best model weights saved to {best_model_path}")
    