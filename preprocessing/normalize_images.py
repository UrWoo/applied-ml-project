import numpy as np
import os
from PIL import Image

def normalize(arr):
    arr_new = (arr / 255) * 2 - 1
    return arr_new

# Define the base project directory using relative paths
base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use relative paths from the project root
folder_path = os.path.join(base_dir, 'data', 'processed_data')
output_path = os.path.join(base_dir, 'data', 'normalized_data')

normalized_images = []

for filename in os.listdir(folder_path):
    img_path = os.path.join(folder_path, filename)
    img = Image.open(img_path).convert('RGB')
    
    img_array = np.array(img).astype(np.float32)
    
    # Normalize the images
    normalized_img = normalize(img_array)
    normalized_images.append(normalized_img)
    
    # Saves the arrays as the normalized images
    # save_img = ((normalized_img + 1) / 2 * 255).astype(np.uint8)
    save_img = normalized_img.astype(np.uint8)
    Image.fromarray(save_img).save(os.path.join(output_path, filename))


normalized_images = np.array(normalized_images)