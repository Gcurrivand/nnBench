import torch
import numpy as np
import os
import sys
package_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, package_path)
from NN import mnist_load, create_labels_array, CNN_V1, cnn_run_multiple_inference
import matplotlib.pyplot as plt

def show_image_from_array(pixel_array):
    plt.imshow(pixel_array, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')  # Hide the axis
    plt.show()

# Load the test data
(x_test, y_test) = mnist_load()[1]

# Convert test data to numpy arrays and then to PyTorch tensors
x_test = np.array(x_test)
y_test = create_labels_array(np.array(y_test))

x_test = torch.tensor(x_test[:, np.newaxis, :, :], dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Load the model once
model = CNN_V1()
model_path = os.path.join(os.path.dirname(__file__), f"best_{model.name}_weights.pth")
model.load_state_dict(torch.load(model_path))
model.eval()

nb_tests = 5000
batch_size = 100  # Adjust the batch size as needed
correct = 0

with torch.no_grad():
    for i in range(0, nb_tests, batch_size):
        batch_x = x_test[i:i+batch_size]
        batch_y = y_test[i:i+batch_size]
        predictions = cnn_run_multiple_inference(model, batch_x)
        correct += (predictions == batch_y.argmax(dim=1)).sum().item()

print(correct / nb_tests * 100)
