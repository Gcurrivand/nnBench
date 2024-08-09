import torch.nn as nn
import torch
from functools import lru_cache
import matplotlib.pyplot as plt
from scipy.ndimage import label, find_objects
import numpy as np
import math

class BBUNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=10):
        super(BBUNET, self).__init__()
        self.name = "bbunet"
        self.ratio = 16

        self.act = nn.Sigmoid()
        self.dwn1 = downSample(in_channels, 64)
        self.dwn2 = downSample(64, 128)
        self.dwn3 = downSample(128, 256)
        self.dwn4 = downSample(256, 512)

        self.bottleneck = doubleConv(512, 1024)

        self.up1 = upSample(1024, 512)
        self.up2 = upSample(512, 256)
        self.up3 = upSample(256, 128)
        self.up4 = upSample(128, 64)
        self.cvend = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)

        self.bboxing = bboxing()

    def forward(self, x):
        x, y1 = self.dwn1(x)
        x, y2 = self.dwn2(x)
        x, y3 = self.dwn3(x)
        x, y4 = self.dwn4(x)
        roi = self.bottleneck(x)
        x = self.up1(roi, y4)
        x = self.up2(x, y3)
        x = self.up3(x, y2)
        x = self.up4(x, y1)
        x = self.cvend(x)
        x = self.act(x)
        bboxes = self.bboxing(x)
        print(bboxes)
        bboxes = rescale_bboxes(bboxes,self.ratio)
        print(bboxes)
        #print(x.shape)
        #display_unet_output(x)
        #display_colored_unet_output(x,2)
        return bboxes

class doubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(doubleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        return out

class downSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(downSample, self).__init__()
        self.dblc = doubleConv(in_channels, out_channels)
        self.maxp1 = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        y = self.dblc(x)
        out = self.maxp1(y)
        return out, y

class upSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(upSample, self).__init__()
        self.trconv1 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels // 3, kernel_size=2, stride=2)
        self.dblc = doubleConv(in_channels // 3 + out_channels, out_channels)

    def forward(self, x, y):
        x = self.trconv1(x)
        out = torch.cat([x, y], dim=1)  # Concatenate along the channel dimension
        out = self.dblc(out)
        return out

class bboxing(nn.Module):
    def __init__(self, initial_min_size=500, initial_max_size=15000):
        super(bboxing, self).__init__()
        # Create learnable parameters
        self.min_group_size = nn.Parameter(torch.tensor(float(initial_min_size)))
        self.max_group_size = nn.Parameter(torch.tensor(float(initial_max_size)))

    def forward(self, x):
        channel_means = x.mean(dim=[2, 3], keepdim=True)
        binary_tensor = (x >= channel_means).int()
        connected = self.find_connected_groups_scipy(binary_tensor)
        return connected

    def find_connected_groups_scipy(self, tensor):
        _, channels, height, width = tensor.shape
        structure = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]], dtype=bool)
        all_bounding_boxes = []
        tensor_np = tensor[0].detach().cpu().numpy()

        # Use the learned parameters
        min_group_size = max(int(self.min_group_size.item()), 1)  # Ensure it's at least 1
        max_group_size = max(int(self.max_group_size.item()), min_group_size + 1)  # Ensure it's greater than min_group_size

        for ch in range(channels):
            channel_data = tensor_np[ch]
            labeled_array, num_features = label(channel_data, structure=structure)
            
            objects = find_objects(labeled_array)
            sizes = np.bincount(labeled_array.ravel())[1:]
            
            valid_indices = np.where((sizes >= min_group_size) & (sizes < max_group_size))[0]
            
            for idx in valid_indices:
                slice_y, slice_x = objects[idx]
                bounding_box = [slice_x.start, slice_y.start, 
                                slice_x.stop - slice_x.start, 
                                slice_y.stop - slice_y.start]
                all_bounding_boxes.append(bounding_box)
        
        return torch.tensor(all_bounding_boxes, dtype=torch.int)

import torch

def rescale_bboxes(bboxes, ratios):
    # Convert bboxes to a tensor if it's a list
    if isinstance(bboxes, list):
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
    
    if isinstance(ratios, int):
        ratios = (ratios, ratios, ratios, ratios)
    
    if len(ratios) != 4:
        raise ValueError("Ratios must be a tuple of four elements (x_ratio, y_ratio, w_ratio, h_ratio)")
    
    # Ensure bboxes is a tensor and get its device
    if not isinstance(bboxes, torch.Tensor):
        raise TypeError("bboxes must be a PyTorch tensor or a list convertible to a tensor")
    
    device = bboxes.device
    
    # Convert ratios to a tensor on the same device as bboxes
    ratios_tensor = torch.tensor(ratios, dtype=torch.float32, device=device)
    
    # Ensure bboxes is a 2D tensor
    if bboxes.dim() == 1:
        bboxes = bboxes.unsqueeze(0)
    
    if bboxes.shape[1] != 4:
        raise ValueError("Each bounding box must have four elements (x, y, w, h)")
    
    # Rescale bboxes and round down
    rescaled_bboxes = torch.div(bboxes.float(), ratios_tensor)
    rescaled_bboxes = torch.floor(rescaled_bboxes).to(torch.int)
    
    return rescaled_bboxes


def display_unet_output(output_tensor):
    print(output_tensor)
    print(output_tensor.shape)
    output_image = output_tensor.detach().cpu().numpy()
    num_channels = output_image.shape[1]
    
    # Iterate over each channel and display the corresponding image
    for i in range(num_channels):
        plt.imshow(output_image[0, i, :, :], cmap='binary', vmin=0, vmax=1)
        plt.title(f"UNet Output - Channel {i}")
        plt.axis('off')
        plt.show()






