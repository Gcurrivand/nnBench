import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from NN import LeNet5, cnn_run_inference
from pygame_image import pygame_rectangle_canvas

canvas_size = 560
image_to_detect_size = 28

def bruteforce_detection(surface):
    for w in range(532):
        for h in range(532):
            pygame_rectangle_canvas(surface,w,h,w+image_to_detect_size,h+image_to_detect_size,"bruteforce.png")
            detected_number = torch.argmax(cnn_run_inference(LeNet5(),"")).item()
            if detected_number in [0,1,2,3,4,5,6,7,8,9]:
                print("detected")

            
