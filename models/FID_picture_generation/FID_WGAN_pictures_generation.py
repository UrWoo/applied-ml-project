import os
import sys
import torch
import torchvision.transforms as transforms

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, project_root)

from models.WGAN_GP import WGAN_GP


results_path = "results"

if not os.path.exists(results_path):
    os.makedirs(results_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = WGAN_GP.Generator(
    noise_dim=100, final_conv_size=128, output_channels=3
)

checkpoint = torch.load(
    "parameters/WGAN-GP-250epochs/generator.pth", map_location=device
)

# Check if the checkpoint contains just the state_dict or is a full checkpoint
if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    generator.load_state_dict(checkpoint["state_dict"])
elif isinstance(checkpoint, dict) and "generator_state_dict" in checkpoint:
    generator.load_state_dict(checkpoint["generator_state_dict"])
else:
    # If checkpoint is just the state_dict itself
    generator.load_state_dict(checkpoint)

generator.to(device)
generator.eval()


# Generate images
num_generate = 1

for i in range(5):
    noise = torch.randn(num_generate, 100, device=device)
    wgan_images_tensor = generator(noise)

    # Normalize the tensor to [0, 1] range and convert to PIL images
    normalized_tensor = (
        wgan_images_tensor[0] + 1
    ) / 2  # Convert from [-1, 1] to [0, 1]
    normalized_tensor = torch.clamp(
        normalized_tensor, 0, 1
    )  # Ensure values are in [0, 1]

    wgan_images_pil = transforms.ToPILImage()(normalized_tensor)
    save_path = f"results/generated_images_ver2/wgan_image_{i}.jpg"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    wgan_images_pil.save(save_path)
    if i % 100 == 0:
        print(f"{i} is finished")
