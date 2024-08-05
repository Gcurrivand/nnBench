import json
import os
import json
from functools import lru_cache
from utils import *

train_path = os.path.join(os.path.join(os.path.dirname(__file__),f"../Dataset/train/_annotations.coco.json"))
valid_path = os.path.join(os.path.join(os.path.dirname(__file__),f"../Dataset/valid/_annotations.coco.json"))


@lru_cache(maxsize=None)
def load_data(path):
    with open(path, 'r') as file:
        return json.load(file)

def get_image_info(data_origin, image_id):
    if data_origin == "train":
        path = train_path
    elif data_origin == "valid":
        path = valid_path
    else:
        return "Invalid data_origin"

    data = load_data(path)

    image_info = next((img for img in data['images'] if img['id'] == image_id), None)
    if not image_info:
        return "Image not found"

    annotations = [ann for ann in data['annotations'] if ann['image_id'] == image_id and not check_bbox(ann['bbox'])]
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

def list_all_categories(data_origin):
    if data_origin == "train":
        path = train_path
    elif data_origin == "valid":
        path = valid_path
    else:
        return "Invalid data_origin"
    data = load_data(path)

    categories = [{"id":cat['id'],"name":cat['name']} for cat in data['categories']]

    return categories
