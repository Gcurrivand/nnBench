from pygame_image import pygame_rectangle_extract
from NN import LeNet5 ,cnn_run_single_inference
import os

canvas_size = 560
image_to_detect_size = 28

def bruteforce_detection(surface):
    current_dir = os.path.dirname(__file__)
    image_path = os.path.join(current_dir, "../Drawings/bruteforce.png")
    for w in range(532):
        for h in range(532):
            pygame_rectangle_extract(surface,w,h,w+image_to_detect_size,h+image_to_detect_size,image_path)
            detected_number = cnn_run_single_inference(LeNet5(),image_path)
            if detected_number in [0,1,2,3,4,5,6,7,8,9]:
               print("detected")
        print(w)

            
