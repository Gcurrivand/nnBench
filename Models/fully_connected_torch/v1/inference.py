import torch
import numpy as np
import os
import sys
package_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, package_path)
from data_package import mnist_load, MLP_V1

# Initialize the model
model = MLP_V1()

# Load the model weights
best_model_path = os.path.join(os.path.dirname(__file__), "best_mlp_v1_weights.pth")
model.load_state_dict(torch.load(model_path))

# Set the model to evaluation mode
model.eval()

# Load the test data
(x_test, y_test) = mnist_load()[1]

# Convert test data to numpy arrays and then to PyTorch tensors
x_test = np.array(x_test)
y_test = np.array(y_test)

x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# No need for gradients during inference
with torch.no_grad():
    predictions = model(x_test).squeeze(1)  # Squeeze to match label shape if necessary

# If you need to round the predictions (since you used a RoundLayer in the model)
rounded_predictions = torch.round(predictions)

# Print the predictions and the actual labels for comparison
for i in range(10):  # Print first 10 predictions as an example
    print(f'Predicted: {rounded_predictions[i].item()}, Actual: {y_test[i].item()}')
