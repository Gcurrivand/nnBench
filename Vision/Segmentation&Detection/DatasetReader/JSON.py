import json
import os

train_path = os.path.join(os.path.join(os.path.dirname(__file__),f"../Dataset/train/_annotations.coco.json"))
valid_path = os.path.join(os.path.join(os.path.dirname(__file__),f"../Dataset/valid/_annotations.coco.json"))

def get_image_info(mode, image_id):
    if mode == "train":
        path = train_path
    elif mode == "valid":
        path = valid_path
    with open(path, 'r') as file:
        data = json.load(file)

    image_info = next((img for img in data['images'] if img['id'] == image_id), None)
    if not image_info:
        return "Image not found"
    annotations = [ann for ann in data['annotations'] if ann['image_id'] == image_id]
    for ann in annotations:
        category = next((cat for cat in data['categories'] if cat['id'] == ann['category_id']), None)
        if category:
            ann['category_name'] = category['name']
            ann['category_supercategory'] = category['supercategory']
    result = {
        "image": image_info,
        "annotations": annotations
    }
    return result