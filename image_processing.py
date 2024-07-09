from PIL import Image
import os

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
    # List of common image file extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    
    # Process each file in the input folder
    for filename in os.listdir(input_folder):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"binarized_{filename}")
            
            try:
                binarize_image(input_path, output_path, threshold)
                print(f"Processed: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

def image_to_bw_array(image_path, threshold=128):
    # Open the image and convert it to grayscale
    with Image.open(image_path) as img:
        img_gray = img.convert('L')
    
    # Convert the image to a numpy array
    img_array = np.array(img_gray)
    
    # Create an array where 1 represents black (below or equal to threshold), 0 represents white
    bw_array = (img_array <= threshold).astype(int)
    
    return bw_array
