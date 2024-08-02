import JSON as j
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import pandas as pd

cfp = os.path.join(os.path.dirname(__file__))
train_path = os.path.join(cfp,f"../Dataset/train")
valid_path = os.path.join(cfp,f"../Dataset/valid")

def write_line_to_csv(filename, param1, param2):
    df = pd.DataFrame({'image_path': [param1], 'category_id': [param2]})
    df.to_csv(filename, data_origin='a', header=not os.path.exists(filename), index=False)

def create_dataset(base_data, nb_base_images, percentage_train):
    data_path = os.path.join(cfp, "../Dataset", "Data")
    os.makedirs(data_path, exist_ok=True)

    train_path = os.path.join(data_path, "Train")
    os.makedirs(train_path, exist_ok=True)
    train_labels_path = os.path.join(train_path, "data.csv")

    valid_path = os.path.join(data_path, "Valid")
    os.makedirs(valid_path, exist_ok=True)
    valid_labels_path = os.path.join(valid_path, "data.csv")

    # Create empty CSV files
    pd.DataFrame(columns=['image_path', 'category_id']).to_csv(train_labels_path, index=False)
    print(f"File 'data.csv' created at {train_labels_path}")
    pd.DataFrame(columns=['image_path', 'category_id']).to_csv(valid_labels_path, index=False)
    print(f"File 'data.csv' created at {valid_labels_path}")

    nb_result_images = 0
    nb_train_images = int(percentage_train * nb_base_images /100)

    for i in range(nb_base_images):
        image_info, image_path = get_image_info_path(base_data, i)
        subs = create_sub(base_data, i, True)
        for j, sub in enumerate(subs):
            if i < nb_train_images:
                save_path = train_path
                save_label = train_labels_path
            else:
                save_path = valid_path
                save_label = valid_labels_path

            resized = resize_image(base_data, i, (28,28), save=True, upscale=True, keep_aspect_ratio=False, image_path=sub)
            gray = convert_to_grayscale(base_data, i, resized, save=True, dest_path=save_path)
            write_line_to_csv(save_label, gray, image_info['annotations'][j]['category_id'])
            nb_result_images += 1
        print(str(i)+" sub images are added to data")

    print(f"Created {nb_result_images} images from {nb_base_images} in {base_data} Dataset")

def contains_non_int(arr):
    return any(not isinstance(item, int) for item in arr)

def too_small_bbox(arr, treshold):
    if arr[2] < treshold:
        return True
    if  arr[3] < treshold:
        return True
    return False

def check_bbox(arr):
    return contains_non_int(arr) or too_small_bbox(arr, 10)

def get_image_info_path(data_origin, image_id):
    image_info = j.get_image_info(data_origin, image_id)
    if data_origin == "train":
        image_path = os.path.join(train_path, image_info['image']['file_name'])
    elif data_origin == "valid":
        image_path = os.path.join(valid_path, image_info['image']['file_name'])
    return image_info,image_path

def read(data_origin, image_id, bbox=False):
    image_info, image_path = get_image_info_path(data_origin, image_id)
    try:
        img = Image.open(image_path)
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        if bbox:
            if check_bbox(bbox):
                print("Bbox is wrong")
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

def draw_bbox(data_origin, image_id, bbox):
    if check_bbox(bbox):
        sys.exit("Bbox is wrong")
    image_info, image_path = get_image_info_path(data_origin, image_id)
    try:
        img = Image.open(image_path)
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        
        # Unpack the bbox coordinates
        x, y, width, height = bbox
        
        # Create and add the rectangle
        rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        
        plt.axis('off')  
        plt.show()
    except Exception as e:
        print(f"An error occurred: {e}")

def show_bbox(data_origin, image_id, bbox):
    if check_bbox(bbox):
        sys.exit("Bbox is wrong")
    image_info, image_path = get_image_info_path(data_origin, image_id)
    try:
        img = Image.open(image_path)
        x, y, width, height = bbox
        cropped_img = img.crop((x, y, x + width, y + height))
        plt.imshow(cropped_img)
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"An error occurred: {e}")

def get_bbox_pixels(data_origin, image_id, bbox):
    if check_bbox(bbox):
        sys.exit("Bbox is wrong")
    image_info, image_path = get_image_info_path(data_origin, image_id)
    try:
        img = Image.open(image_path)
        x, y, width, height = bbox
        cropped_img = img.crop((x, y, x + width, y + height))
        return cropped_img
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def convert_to_grayscale(data_origin, image_id, image_path=None, dest_path="../Dataset/grayscaled_images", save=False):
    dest_dir = os.path.join(cfp,dest_path)
    os.makedirs(dest_dir, exist_ok=True)
    if not image_path:
        image_info, image_path = get_image_info_path(data_origin, image_id)
    try:
        with Image.open(image_path) as img:
            grayscale_img = img.convert('L')
            grayscale_image_filename = f"gray_{os.path.basename(image_path)}"
            save_path = os.path.join(dest_dir, grayscale_image_filename)
            if save:
                grayscale_img.save(save_path)
            return save_path
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def resize_image(data_origin, image_id, target_size, keep_aspect_ratio=True, upscale=False, save=False, dest_path="../Dataset/resized_images", image_path=None):
    dest_dir = os.path.join(cfp,dest_path)
    os.makedirs(dest_dir, exist_ok=True)
    if not image_path:
        image_info, image_path = get_image_info_path(data_origin, image_id)
    with Image.open(image_path) as img:
        if img.data_origin == 'P':
            img = img.convert('RGB')

        original_width, original_height = img.size
        target_width, target_height = target_size

        if keep_aspect_ratio:
            scale = min(target_width / original_width, target_height / original_height)
            if not upscale:
                scale = min(scale, 1.0)   
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
        else:
            new_width, new_height = target_width, target_height        
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)

        if keep_aspect_ratio:
            new_img = Image.new('RGB', (target_width, target_height), (255, 255, 255))
            paste_x = (target_width - new_width) // 2
            paste_y = (target_height - new_height) // 2
            new_img.paste(resized_img, (paste_x, paste_y))
            resized_image_filename = f"{image_id}_resized_{os.path.basename(image_path)}"
            save_path = os.path.join(dest_dir, resized_image_filename)
            if save:
                new_img.save(save_path)
            return save_path
        else:
            resized_image_filename = f"{image_id}_resized_{os.path.basename(image_path)}"
            save_path = os.path.join(dest_dir, resized_image_filename)
            if save:
                resized_img.save(save_path)
            return save_path

def create_sub(data_origin, image_id, save=False, path="../Dataset/sub_images"):
    image_info, image_path = get_image_info_path(data_origin, image_id)
    saved_images = []
    
    try:
        with Image.open(image_path) as img:
            for i, annotation in enumerate(image_info['annotations']):
                bbox = annotation['bbox']
                if check_bbox(bbox):
                    print("Bbox is wrong")
                    continue
                category_name = annotation['category_name']
                x, y, width, height = bbox
                img_width, img_height = img.size
                x = max(0, x)
                y = max(0, y)
                right = min(img_width, x + width)
                bottom = min(img_height, y + height)
                bbox_image = img.crop((x, y, right, bottom))
                sub_image_filename = f"{image_id}_{str(category_name).replace(" ","_")}_{i}.png"
                save_path = os.path.join(cfp,path, sub_image_filename)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                if save:
                    bbox_image.save(save_path)
                saved_images.append(save_path)
        return saved_images
    except Exception as e:
        print(f"An error occurred while extracting and saving the bounding box images: {e}")
        return None