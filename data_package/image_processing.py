from PIL import Image
import os
import numpy as np

def binarize_image(input_path, output_path, threshold=128):
    with Image.open(input_path) as img:
        print(input_path)
        gray_img = img.convert('L')
        # Apply threshold
        binary_img = gray_img.point(lambda x: 0 if x < threshold else 255, '1')
        binary_img.save(output_path)

def binarize_folder(input_folder, output_folder, threshold=128): 
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']  
    for filename in os.listdir(input_folder):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"binarized_{filename}")
            
            try:
                binarize_image(input_path, output_path, threshold)
                print(f"Processed: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

def bw_image_to_2d_array(image_path, threshold=128):
    with Image.open(image_path) as img:
        img_gray = img.convert('L')
    img_array = np.array(img_gray)
    bw_array = (img_array <= threshold).astype(int)
    return bw_array

def bw_image_to_1d_array(image_path, threshold=128):
    img = Image.open(image_path).convert('L')
    width, height = img.size
    result = np.zeros(width * height, dtype=np.int8)
    for i, pixel in enumerate(img.getdata()):
        result[i] = 1 if pixel < threshold else 0
    return result
