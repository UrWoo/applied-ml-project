import os
import random
import shutil

source = "data/processed_data/images"
destination = "data/FID_images"

if not os.path.exists(destination):
    os.makedirs(destination)

images = [i for i in os.listdir(source)]

selected = random.sample(images, 20000)

for image in selected:
    shutil.move(os.path.join(source, image), os.path.join(destination, image))
