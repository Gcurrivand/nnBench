import os
import numpy as np
from .image_processing import bw_image_to_1d_array


def create_data_from_bw_images(input_folder, threshold=128): 
    result = []
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']  
    input_folder = os.path.abspath(input_folder)
    for filename in os.listdir(input_folder):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            input_path = os.path.join(input_folder, filename)
            try:
                result.append(bw_image_to_1d_array(input_path, threshold))
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    return np.array(result, dtype=int)

def read_labels(folder_path):
    contents = []
    if not os.path.isdir(folder_path):
        return np.array([f"Error: The folder '{folder_path}' does not exist."])
    folder_path = os.path.abspath(folder_path)
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    contents.append(int(file.read()))
            except Exception as e:
                contents.append(f"Error reading file {filename}: {str(e)}")
    return np.array(contents, dtype=int)

def create_labels_array(original_array):
    # Ensure the original array is of the correct type and shape
    original_array = np.asarray(original_array, dtype=int)
    if np.any((original_array < 0) | (original_array > 9)):
        raise ValueError("Input must be a 1D array of length 10 with values between 0 and 9")
    result = []
    for value in original_array:
        result.append(create_label_array(value))
    return np.array(result)

def create_label_array(value):
    new_array = np.zeros(10, dtype=int)
    new_array[value] = 1
    return np.array(new_array)