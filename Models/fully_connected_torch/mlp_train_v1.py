import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import numpy as np
from torch.utils.data import Dataset, DataLoader
from MLP import MLP_V1
package_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, package_path)
from data_package import read_labels, create_data_from_bw_images, create_indicator_array, mnist_load

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
batch_size = 3000
image_size = 28
lr = 0.95
# Ensure the code runs only when the script is executed directly
if __name__ == '__main__':
    dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=0)
    model = MLP_V1(image_size, lr).to("cpu")
    
    # Training loop
    num_epochs = 200
    for epoch in range(num_epochs):
        for batch_idx, (data, labels) in enumerate(dataloader):
            # No need to unsqueeze the labels
            
            # Forward pass
            outputs = model(data).squeeze(1)  # Squeeze the output to match the label shape
            loss = model.criterion(outputs, labels)
            
            # Backward and optimize
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    print("Training complete.")
    
    # Save the model's weights
    model_path = "mlp_v1_weights.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model weights saved to {model_path}")
