from JSON import get_image_info
from CRUD import read, create_sub, draw_bbox, resize_image, convert_to_grayscale, create_dataset
from CUSTOMDATASET import CustomDataset
import os
import sys
package_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, package_path)
from NN import LeNet5, train, cnn_run_single_inference


cfp = os.path.join(os.path.dirname(__file__))

import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    # Your existing code here
    #train(LeNet5().to("cpu"), 100, 3000)
    cnn_run_single_inference(LeNet5().to("cpu"),os.path.join(cfp,"../Dataset/Data/Valid/gray_4084_resized_4084_person_6.png"))

#print(get_image_info("train", 96))
#subs = create_sub("train", 96, True)
#print(subs)
#read("train", 1)
#draw_bbox("train", 1,[10,10,20,20])
#for i in range(1, 100):
#    print(i)
#    subs = create_sub("train", i, True)
#    for sub in subs:
#       resized = resize_image("train", i, (60,60), save=True, upscale=True, keep_aspect_ratio=False,image_path=sub)
#       convert_to_grayscale("train", i, resized, save=True)
#start = time.time()
#create_dataset("train", 5000, 80)
#end = time.time()
#print(end-start)