from JSON import get_image_info
from CRUD import read, create_sub, draw_bbox, resize_image, convert_to_grayscale, create_dataset, show_bbox, get_image_info_path
from CUSTOMDATASET import CustomDataset
import os
import sys
package_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, package_path)
from NN import LeNet5, train, cnn_run_single_inference_image_pil, create_random_kernels, image_slide_kernel_function
import pandas as pd

cfp = os.path.join(os.path.dirname(__file__))

import multiprocessing
import random

def validate_selection(probability):
    return random.random() < probability

""" if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    datas = pd.read_csv(os.path.join(cfp, "../Dataset/Data/Train", "data.csv"))
    rows_to_remove = []
    print(len(datas))
    for i in range(len(datas)):
        category_id = datas.iloc[i]["category_id"]
        image_path = os.path.join(os.path.join(cfp, "../Dataset/Data/Train"), datas.iloc[i]["image_path"])
        should_delete = False
        
        if category_id == 19 or category_id == 23:
           should_delete = validate_selection(1/2)
        
        if should_delete:
            # Delete the image file
            if os.path.exists(image_path):
                print(image_path)
                os.remove(image_path)
            # Mark this row for removal from the DataFrame
            rows_to_remove.append(i)

    # Remove marked rows from the DataFrame
    datas = datas.drop(datas.index[rows_to_remove])

    # Save the updated DataFrame back to the CSV file
    datas.to_csv(os.path.join(cfp, "../Dataset/Data/Train", "data.csv"), index=False) """
    
    

