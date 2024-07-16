import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import numpy as np
from torch.utils.data import Dataset, DataLoader
package_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, package_path)
from data_package import mnist_load, MLP_V1

(data, labels), (x_test, y_test) = mnist_load()

# Convert data to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Convert data to PyTorch tensors
data = torch.tensor(data, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.float32)

# Define a custom dataset
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label

# Instantiate the dataset
dataset = CustomDataset(data, labels)
# Ensure the code runs only when the script is executed directly
if __name__ == '__main__':
    model = MLP_V1().to("cpu")
    dataloader = DataLoader(dataset, model.batch_size, shuffle=True, num_workers=0)
    
    best_loss = float('inf')
    best_model_path = os.path.join(os.path.dirname(__file__), "best_mlp_v1_weights.pth")
    num_epochs = 100
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, (data, labels) in enumerate(dataloader):
            outputs = model(data).squeeze(1)
            loss = model.criterion(outputs, labels)
            
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with loss: {best_loss:.4f}")
    
    print("Training complete.")
    print(f"Best model weights saved to {best_model_path}")
