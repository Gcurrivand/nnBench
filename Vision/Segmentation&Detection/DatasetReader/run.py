from JSON import *
from CRUD import *
from CUSTOMDATASET import CustomDataset
from Visualization import *
import os
import sys
package_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, package_path)
from NN import *


cfp = os.path.join(os.path.dirname(__file__))

import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()
    #print(show_bar_graph(add_count_class_occurrences(os.path.join(cfp,"../Dataset/Resnet50/Train"),list_all_categories("train"))))
    #train(VGG16(lr=0.1, momentum=0.9).to("cpu"), 10, 500,data_path=os.path.join(cfp,"../Dataset/Resnet50/Train"))
    #image_name = "gray_9534_resized_9534_sandwich_0.png"
    #cnn_run_single_inference_image_path(LeNet5().to("cpu"),os.path.join(cfp,"../Dataset/Data/Valid",image_name))
    #"C:/Users/Currivandg/Downloads/Futur/nnBench\Vision/Segmentation&Detection/Dataset/train/000000010037_jpg.rf.df2ea3f8c29692d78e208268f7278b17.jpg"
    info , path = get_image_info_path("train", 86271)
    print(info)
    bboxes = [data['bbox'] for data in info['annotations']]
    print(bboxes)
    ratio = compute_ratio((512,512), (info['image']['width'],info['image']['height']))
    print(ratio)
    rescaled = rescale_bboxes(bboxes,(ratio[0],ratio[1],ratio[0],ratio[1]))
    print(rescaled)
    resized = resize_image((512,512), save=False, image_path=path)
    input_image = PILImageToTensor(resized).unsqueeze(0)
    model = BBUNET(out_channels=50)
    output = model(input_image)
    print(output)
    draw_bboxes("train", 86271, output.numpy(), img=resized)
    draw_bboxes("train", 86271, bboxes=rescaled, img=resized,color='g')
    #display_unet_output(output)
    #display_colored_unet_output(output,3)
    # Example usage:
    # Let's say you have a 2D numpy array representing an image
    # Example usage:
    # Let's say you have a 2D numpy array representing an image

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
#create_dataset("train", 10000, 95, size=224, name="Resnet50")
#end = time.time()
#print(end-start)