import os
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder, MNIST
import torchvision.utils
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from DCGAN import Discriminator, Generator, init_weights

# Set up hyperparameters
data_root = "processed_data"
batch_size = 128
noise_dimension = 100
final_convolution_classes = 128
image_channels = 3
learning_rate = 0.0002
beta_one = 0.5
epochs = 50

results_path = 'results'
grid_path = 'results/grid'

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
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)

# Set up device to use accelerator when available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up generator
generator = Generator(
    noise_dim=noise_dimension,
    final_conv_size=final_convolution_classes,
    output_channels=image_channels,
).to(device)

generator.apply(init_weights)

# Set up discriminator
discriminator = Discriminator(
    first_conv_size=final_convolution_classes,
    input_channels=image_channels,
).to(device)

discriminator.apply(init_weights)

# Set up optimizers
gen_opt = torch.optim.Adam(
    generator.parameters(), lr=learning_rate, betas=(beta_one, 0.999)
)
disc_opt = torch.optim.Adam(
    discriminator.parameters(), lr=learning_rate, betas=(beta_one, 0.999)
)

# Set up loss function
loss = nn.BCELoss()

# Fixed noise vector to visualize training
fixed_noise = torch.rand(64, noise_dimension).to(device) * 2 - 1

gen_losses = []
disc_losses = []
images = []

for epoch in range(epochs):
    running_gen_loss = 0
    running_disc_loss = 0
    for i, data in enumerate(dataloader):
        # Train discriminator
        # Real images
        real_images = data[0].to(device)
        disc_real_output = discriminator(real_images).flatten()
        disc_real_labels = torch.ones_like(disc_real_output)
        disc_real_loss = loss(disc_real_output, disc_real_labels)

        # Generate noise from uniform distribution
        noise = (
            torch.rand(real_images.size(0), noise_dimension).to(device) * 2 - 1
        )

        # Fake images
        fake_images = generator(noise)
        disc_fake_output = discriminator(fake_images.detach()).flatten()
        disc_fake_labels = torch.zeros_like(disc_fake_output)
        disc_fake_loss = loss(disc_fake_output, disc_fake_labels)

        # Calculate loss
        disc_loss = disc_real_loss + disc_fake_loss

        # Do a backwards pass
        disc_opt.zero_grad()
        disc_loss.backward()
        disc_opt.step()

        # Train generator
        gen_disc_output = discriminator(fake_images).flatten()
        gen_loss = loss(gen_disc_output, disc_real_labels)
        
        # Do a backwards pass
        gen_opt.zero_grad()
        gen_loss.backward()
        gen_opt.step()

        # Calculate running loss
        running_gen_loss += gen_loss.item() * data[0].size(0)
        running_disc_loss += disc_loss.item() * data[0].size(0)

    with torch.no_grad():
        imgs = generator(fixed_noise).detach()
    images.append(imgs)

    gen_losses.append(running_gen_loss / len(dataloader.dataset))
    disc_losses.append(running_disc_loss / len(dataloader.dataset))


# Print loss graph
plt.title("Generator and Discriminator Losses")
plt.plot(gen_losses,label="generator losses")
plt.plot(disc_losses,label="discriminator losses")
plt.xlabel("i")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(results_path, 'loss_graph.png'))
plt.close()

# Print progress on a fixed noise vector
for i, img in enumerate(images):
    grid = torchvision.utils.make_grid(img, padding=2, normalize=True)

    plt.figure()
    plt.imshow(grid.permute(1,2,0).cpu().numpy())
    plt.axis('off')
    plt.savefig(os.path.join(grid_path, f'grid_epoch_{i}.png'))
    plt.close()

torch.save(generator.state_dict(), 'generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')