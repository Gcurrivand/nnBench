import torch
import numpy as np
import os
import sys
package_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, package_path)
from NN import mnist_load, create_labels_array, LeNet5, cnn_run_inference
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

x_test = torch.tensor(x_test[ :, np.newaxis, :, :], dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

nb_tests = 5000
correct = 0
for i in range(nb_tests):  
    with torch.no_grad():
        rounded_prediction = torch.round(cnn_run_inference(LeNet5(),x_test[i])).squeeze(0)
        if torch.equal(rounded_prediction,y_test[i]):
            correct +=1
        else:
            print(i)
            print(rounded_prediction)
            print(y_test[i])

print(correct/nb_tests*100)
