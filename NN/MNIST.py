import numpy as np
import torch
import struct
import os
from array import array
from PIL import Image


#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)  

def mnist_load():
    input_path = 'Dataset/mnist'
    training_images_filepath =  os.path.abspath(input_path +'/train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = os.path.abspath(input_path + '/train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = os.path.abspath(input_path + '/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = os.path.abspath(input_path + '/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')
    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    return mnist_dataloader.load_data()

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

def load_and_prepare_image(image_path):
    image = Image.open(image_path).convert('L')  # Convert to grayscale ('L' mode)
    if image.size != (28, 28):
        raise ValueError("Image must be 28x28 in size.")
    image_array = np.array(image)
    image_tensor = torch.tensor(image_array, dtype=torch.float32)
    return image_tensor