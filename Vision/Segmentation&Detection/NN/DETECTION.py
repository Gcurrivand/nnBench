import numpy as np
import os
import sys
package_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, package_path)
from DatasetReader import get_image_info, get_bbox_pixels, get_image_info

def create_random_kernels(nb, min_width=30, min_height=30, max_width=300, max_height=300, rate=0.7):
    kernels = []
    for _ in range(nb):
        while True:
            width = np.random.randint(min_width, max_width)
            height = np.random.randint(min_height, max_height)
            if abs(width - height) / max(width, height) <= rate:
                kernels.append([width, height])
                break
    return np.array(kernels)

def image_slide_kernel_function(kernel, data_origin,image_id, function, stride_x=1, stride_y=1):
    if not function:
        sys.exit("Function need to be provided to be applied for each iteration of kernel")
    img_info = get_image_info(data_origin, image_id)
    image_width = img_info["image"]["width"]
    image_height = img_info["image"]["width"]
    kernel_width, kernel_height = kernel
    num_positions_x = (image_width - kernel_width) // stride_x + 1
    num_positions_y = (image_height - kernel_height) // stride_y + 1
    for y in range(0, num_positions_y * stride_y, stride_y):
        for x in range(0, num_positions_x * stride_x, stride_x):
            bbox = [x,y,kernel_width,kernel_height]
            img = get_bbox_pixels(data_origin, image_id, bbox)
            function(img=img)
            a = input("miaou")