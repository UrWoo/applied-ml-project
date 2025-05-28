import random_model
import torchvision.transforms as transforms
import os
import torch


results_path = "results"

if not os.path.exists(results_path):
    os.makedirs(results_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_images = 20000

noise = torch.randn(num_images, 100, device=device)
generator = random_model.Generator(100, 128, 3)

num_generate = 1
generator.to(device)
generator.eval()

for i in range(5):
    noise = torch.randn(num_generate, 100, device=device)
    generator.apply(random_model.init_weights)

    random_image_tensor = generator(noise)
    normalized_tensor = (random_image_tensor[0] + 1) / 2
    normalized_tensor = torch.clamp(normalized_tensor, 0, 1)

    random_images_pil = transforms.ToPILImage()(normalized_tensor)
    save_path = f"results/random_images/random_image_{i}.jpg"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    random_images_pil.save(save_path)
    if i % 100 == 0:
        print(f"{i} is finished")
