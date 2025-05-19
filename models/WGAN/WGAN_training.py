import os
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder, MNIST
import torchvision.utils
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from WGAN import critic, Generator, init_weights

# Set up hyperparameters
data_root = "processed_data"
batch_size = 64
noise_dimension = 100
final_convolution_classes = 128
image_channels = 3
learning_rate = 0.00005
beta_one = 0.5
epochs = 250
critic_iterations = 5
clipping_parameter = 0.01

results_path = "results"
grid_path = "results/grid"

if not os.path.exists(results_path):
    os.makedirs(results_path)
if not os.path.exists(grid_path):
    os.makedirs(grid_path)


# Load dataset
dataset = ImageFolder(
    root=data_root,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    ),
)

# Set up dataloader
dataloader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=4,
)

# Set up device to use accelerator when available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up generator
generator = Generator(
    noise_dim=noise_dimension,
    final_conv_size=final_convolution_classes,
    output_channels=image_channels,
).to(device)

generator.apply(init_weights)

# Set up critic
critic = critic(
    first_conv_size=final_convolution_classes,
    input_channels=image_channels,
).to(device)

critic.apply(init_weights)

# Set up optimizers
gen_opt = torch.optim.RMSprop(generator.parameters(), lr=learning_rate)
critic_opt = torch.optim.RMSprop(critic.parameters(), lr=learning_rate)

# Fixed noise vector to visualize training
fixed_noise = torch.rand(64, noise_dimension).to(device) * 2 - 1

gen_losses = []
critic_losses = []
images = []
data_iter = iter(dataloader)
i = 0

for epoch in range(epochs):
    # TODO change to until converges
    # Train critic
    for _ in range(critic_iterations):
        # Sample batch from real data
        try:
            real_images = next(data_iter).to(device)
        except StopIteration:
            data_iter = iter(dataloader)
            real_images = next(data_iter).to(device)

        # Generate noise from uniform distribution
        noise = (
            torch.rand(real_images.size(0), noise_dimension).to(device) * 2 - 1
        )

        # Get batch from generator
        fake_images = generator(noise)

        # Get critic predictions
        critic_real_images = critic(real_images).flatten()
        critic_fake_images = critic(fake_images).flatten()

        # Get loss
        critic_loss = -(
            torch.mean(critic_real_images) - torch.mean(critic_fake_images)
        )

        # Do a backwards pass
        critic.zero_grad()
        critic_loss.backward()
        critic_opt.step()

        # Clip weights
        for param in critic.parameters():
            param.data.clamp_(-clipping_parameter, clipping_parameter)

    # Train generator

    # Generator noise
    noise = torch.rand(real_images.size(0), noise_dimension).to(device) * 2 - 1

    # Generate generator outpus
    fake_images = generator(noise)

    # Get critic prediction
    critic_fake_images = (fake_images).flatten()

    # Get loss
    gen_loss = -torch.mean(critic_fake_images)

    # Do a backwards pass
    generator.zero_grad()
    gen_loss.backward()
    gen_opt.step()

    # Log losses
    gen_losses.append(gen_loss.item())
    critic_losses.append(critic_loss.item())

    if i % 500 == 0:
        with torch.no_grad():
            imgs = generator(fixed_noise).detach()
        images.append(imgs)
    i += 1


# Print loss graph
plt.title("Generator and critic Losses")
plt.plot(gen_losses, label="generator losses")
plt.plot(critic_losses, label="critic losses")
plt.xlabel("i")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(results_path, "loss_graph.png"))
plt.close()

# Print progress on a fixed noise vector
for i, img in enumerate(images):
    grid = torchvision.utils.make_grid(img, padding=2, normalize=True)

    plt.figure()
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis("off")
    plt.savefig(os.path.join(grid_path, f"grid_epoch_{i}.png"))
    plt.close()

torch.save(generator.state_dict(), "generator.pth")
torch.save(critic.state_dict(), "critic.pth")
