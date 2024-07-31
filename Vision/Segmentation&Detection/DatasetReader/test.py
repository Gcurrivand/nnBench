from JSON import get_image_info
from CRUD import read, create_sub, draw_bbox, resize_image, convert_to_grayscale, create_dataset

#print(get_image_info("train", 7))
#read("train", 1)
#draw_bbox("train", 1,[10,10,20,20])
#for i in range(1, 100):
#    print(i)
#    subs = create_sub("train", i, True)
#    for sub in subs:
#       resized = resize_image("train", i, (60,60), save=True, upscale=True, keep_aspect_ratio=False,image_path=sub)
#       convert_to_grayscale("train", i, resized, save=True)
create_dataset("train", 10, 70)