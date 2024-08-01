import json
import os
import json
from functools import lru_cache

train_path = os.path.join(os.path.join(os.path.dirname(__file__),f"../Dataset/train/_annotations.coco.json"))
valid_path = os.path.join(os.path.join(os.path.dirname(__file__),f"../Dataset/valid/_annotations.coco.json"))


@lru_cache(maxsize=None)
def load_data(path):
    with open(path, 'r') as file:
        return json.load(file)

def get_image_info(mode, image_id):
    if mode == "train":
        path = train_path
    elif mode == "valid":
        path = valid_path
    else:
        return "Invalid mode"

    data = load_data(path)

    image_info = next((img for img in data['images'] if img['id'] == image_id), None)
    if not image_info:
        return "Image not found"

    annotations = [ann for ann in data['annotations'] if ann['image_id'] == image_id]
    categories = {cat['id']: cat for cat in data['categories']}

    for ann in annotations:
        category = categories.get(ann['category_id'])
        if category:
            ann['category_name'] = category['name']
            ann['category_supercategory'] = category['supercategory']

    result = {
        "image": image_info,
        "annotations": annotations
    }
    return result
