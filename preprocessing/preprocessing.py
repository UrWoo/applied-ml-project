import torch
import torchvision.transforms as transforms
from PIL import Image
import os

base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

folder_path = os.path.join(base_dir, 'data', 'CelebFaces Attributes Dataset', 'img_align_celeba', 'img_align_celeba')
output_path = os.path.join(base_dir, 'data', 'processed_data', 'images')

if not os.path.exists(output_path):
    os.makedirs(output_path)

transform = transforms.Resize((64, 64))

for filename in os.listdir(folder_path):
    img_path = os.path.join(folder_path, filename)
    img = Image.open(img_path)
    resized_img = transform(img)
    resized_img.save(os.path.join(output_path, filename))
