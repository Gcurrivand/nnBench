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

class ResNet18(nn.Module):
    def __init__(self, nb_classes, in_channels=3, lr=0.1, momentum=0.9):
        super(ResNet18, self).__init__()
        self.name = "resnet18"

        self.cl1 = nn.Conv2d(kernel_size=(7,7), in_channels=in_channels, out_channels=64, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxp1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 2, 64, 64)
        self.layer2 = self._make_layer(BasicBlock, 2, 64, 128, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 2, 128, 256, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 2, 256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, nb_classes)

        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.cl1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxp1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    def _make_layer(self, block, num_blocks, in_channels, out_channels, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample))
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out
    
class VGG16(nn.Module):
    def __init__(self, lr= 0.1, momentum= 0.9):
        super(VGG16, self).__init__()
        self.name = "vgg16"
        self.lr = lr
        self.momentum = momentum

        self.relu = nn.ReLU()
        self.sofm = nn.Softmax(dim=1)
        self.flat = nn.Flatten()
        self.layer1 = VGGBlock(3,64)
        self.layer2 = VGGBlock(64,128)
        self.layer3 = VGGBlock(128,256,True)
        self.layer4 = VGGBlock(256,512,True)
        self.layer5 = VGGBlock(512,512,True)
        self.fc1 = nn.Linear(25088,4096)
        self.fc2 = nn.Linear(4096,81)

        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        print(x.shape)
        x = self.layer1(x)
        print(x.shape)
        x = self.layer2(x)
        print(x.shape)
        x = self.layer3(x)
        print(x.shape)
        x = self.layer4(x)
        print(x.shape)
        x = self.layer5(x)
        print(x.shape)
        x = self.flat(x)
        print(x.shape)
        x = self.fc1(x)
        print(x.shape)
        x = self.relu(x)
        print(x.shape)
        x = self.fc2(x)
        print(x.shape)
        x = self.relu(x)
        print(x.shape)
        x = self.sofm(x)
        return x
    
class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, lastconv=False):
        super(VGGBlock, self).__init__()
        self.lastconv = lastconv
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.maxp1 = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        print("VGG")
        print(x.shape)
        x = self.conv1(x)
        print(x.shape)
        x = self.relu(x)
        print(x.shape)
        x = self.conv2(x)
        print(x.shape)
        if self.lastconv:
            x = self.conv2(x)
        print(x.shape)
        x = self.relu(x)
        print(x.shape)
        x = self.maxp1(x)
        print(x.shape)
        return x
    


def cnn_compute_inference(model, tensor_input, weights_path=None):
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
    return cnn_compute_inference(model, tensor_input, weights_path)

def cnn_run_single_inference_image_pil(model, img, weights_path=None):
    if not weights_path:
        weights_path = f"./Weights/best_{model.name}_weights.pth"
    tensor_input = PILImageToTensor(img).unsqueeze(0)
    return cnn_compute_inference(model, tensor_input, weights_path)

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
    