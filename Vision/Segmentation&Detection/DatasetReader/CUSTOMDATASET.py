from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms
from PIL import Image
import torch
import os
import random
from functools import lru_cache

nb_classes = 81

def validate_selection(probability):
    return random.random() < probability

@lru_cache(maxsize=None)
def PathImageToTensor(path, normalize=False):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if normalize else lambda x: x
    ])
    img = Image.open(path).convert('RGB')
    tensor = transform(img)
    return tensor

def PILImageToTensor(img, normalize=False):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if normalize else lambda x: x
    ])
    tensor = transform(img)
    return tensor

def create_tensor_with_index_one(size, index):
    tensor = torch.zeros(size)  
    tensor[index] = 1
    return tensor

class CustomDataset(Dataset):
    def __init__(self, csv_file, image_dir):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        print(f"Data has: {len(self.data)} entries")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.image_dir, row["image_path"])
        tensor_img = PathImageToTensor(image_path, True)
        class_id = row["category_id"]
        return tensor_img, class_id