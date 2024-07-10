import torch
import os
import sys
from MLP import MLP_V1

package_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, package_path)
from data_package import bw_image_to_1d_array

def prepare_input(image_path):
    image_array = bw_image_to_1d_array(image_path)
    image_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0)
    return image_tensor

def run_inference(image_path):
    model = MLP_V1().to("cpu")
    model.eval()
    with torch.no_grad():
        input_tensor = prepare_input(image_path)
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        _, predicted = torch.max(output.data, 1)
        return predicted.item(), probabilities.squeeze().tolist()

if __name__ == '__main__':
    test_image_path = "./Dataset/NumbersFC/binarized_drawing_21.png"
    predicted_class, class_probabilities = run_inference(test_image_path)
    print(f"Predicted class: {predicted_class}")
    print(f"Class probabilities: {class_probabilities}")
