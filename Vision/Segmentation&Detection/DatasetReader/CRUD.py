import JSON as j
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

cfp = os.path.join(os.path.dirname(__file__))
train_path = os.path.join(cfp,f"../Dataset/train")
valid_path = os.path.join(cfp,f"../Dataset/valid")

def get_image_path(mode, image_id):
    image_info = j.get_image_info(mode, image_id)
    if mode == "train":
        image_path = os.path.join(train_path, image_info['image']['file_name'])
    elif mode == "valid":
        image_path = os.path.join(valid_path, image_info['image']['file_name'])
    return image_info,image_path

def read(mode, image_id, bbox=False):
    image_info, image_path = get_image_path(mode, image_id)
    
    try:
        img = Image.open(image_path)
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        if bbox:
            for annotation in image_info['annotations']:
                bbox = annotation['bbox']
                category_name = annotation['category_name']
                x, y, width, height = bbox
                rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')
                ax.text(x, y, category_name, color='r', fontsize=8, 
                    bbox=dict(facecolor='white', edgecolor='red', alpha=0.7, pad=0))
                ax.add_patch(rect)
        plt.axis('off')  
        plt.show()
    except Exception as e:
        print(f"An error occurred: {e}")

def create_sub(mode, image_id, save=False):
    image_info, image_path = get_image_path(mode, image_id)
    saved_images = []
    
    try:
        with Image.open(image_path) as img:
            for i, annotation in enumerate(image_info['annotations']):
                bbox = annotation['bbox']
                category_name = annotation['category_name']
                x, y, width, height = bbox
                img_width, img_height = img.size
                x = max(0, x)
                y = max(0, y)
                right = min(img_width, x + width)
                bottom = min(img_height, y + height)
                bbox_image = img.crop((x, y, right, bottom))
                sub_image_filename = f"{image_id}_{category_name}_{i}.png"
                save_path = os.path.join(cfp,"../Dataset/sub_images", sub_image_filename)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                if save:
                    bbox_image.save(save_path)
                saved_images.append(save_path)
        return saved_images
    except Exception as e:
        print(f"An error occurred while extracting and saving the bounding box images: {e}")
        return None