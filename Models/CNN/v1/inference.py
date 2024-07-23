import torch
import numpy as np
import os
import sys
package_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, package_path)
from NN import mnist_load, create_labels_array, CNN_V1
import matplotlib.pyplot as plt


def show_image_from_array(pixel_array):
    plt.imshow(pixel_array, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')  # Hide the axis
    plt.show()

# Initialize the model
model = CNN_V1()

# Load the model weights
model_path = os.path.join(os.path.dirname(__file__), "best_cnn_v1_weights.pth")
model.load_state_dict(torch.load(model_path))

# Set the model to evaluation mode
model.eval()

# Load the test data
(x_test, y_test) = mnist_load()[1]

# Convert test data to numpy arrays and then to PyTorch tensors
x_test = np.array(x_test)
y_test = create_labels_array(np.array(y_test))

x_test = torch.tensor(x_test[ :, np.newaxis, :, :], dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# No need for gradients during inference
with torch.no_grad():
    predictions = model(x_test).squeeze(1)  # Squeeze to match label shape if necessary

# If you need to round the predictions (since you used a RoundLayer in the model)
rounded_predictions = torch.round(predictions)

nb_tests = 5000
correct = 0
for i in range(nb_tests):  
    if(torch.equal(rounded_predictions[i], y_test[i])):
        correct +=1
    else:
        print(i)
        print(rounded_predictions[i])
        print(y_test[i])
    
print(correct/nb_tests*100)
