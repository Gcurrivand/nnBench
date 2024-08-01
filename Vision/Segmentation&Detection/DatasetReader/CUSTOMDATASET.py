from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms
from PIL import Image
import torch
import os

nb_classes = 81

def ImageToTensor(path):
    transform = transforms.Compose([transforms.PILToTensor()])
    img = Image.open(path)
    tensor = transform(img)
    return tensor.float()

def create_tensor_with_index_one(size, index):
    tensor = torch.zeros(size)  
    tensor[index] = 1
    return tensor

class CustomDataset(Dataset):
    def __init__(self, csv_file, image_dir):
        datas = pd.read_csv(csv_file)
        self.data = []
        for i in range(len(datas)):
            image_path = os.path.join(image_dir, datas.iloc[i]["image_path"])
            category_id = datas.iloc[i]["category_id"]
            self.data.append((ImageToTensor(image_path), category_id))
        self.image_dir = image_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tensor_img = self.data[idx][0]
        class_id = self.data[idx][1]
        return tensor_img, class_id
