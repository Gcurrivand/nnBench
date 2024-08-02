from JSON import get_image_info
from CRUD import read, create_sub, draw_bbox, resize_image, convert_to_grayscale, create_dataset, show_bbox, get_image_info_path
from CUSTOMDATASET import CustomDataset
import os
import sys
package_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, package_path)
from NN import LeNet5, train, cnn_run_single_inference_image_pil, create_random_kernels, image_slide_kernel_function


cfp = os.path.join(os.path.dirname(__file__))

import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()
    #show_bbox("train", 1, [50,50,50,50])
    #print(create_random_kernels(3))
    image_slide_kernel_function([50,50],"train", 1, cnn_run_single_inference_image_pil(LeNet5().to("cpu")))
