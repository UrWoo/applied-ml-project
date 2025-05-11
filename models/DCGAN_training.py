import os
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.utils
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from DCGAN import Discriminator, Generator, init_weights

# Set up hyperparameters
data_root = ""
batch_size = 128
noise_dimension = 100
final_convolution_classes = 128
image_channels = 3
learning_rate = 0.0002
beta_one = 0.5
epochs = 1


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
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# Set up device to use accelerator when available
device = torch.device(
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)

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

for epoch in range(epochs):
    for i, data in enumerate(dataloader):
        print(epoch, i)
        if i > 50:
            break
        # Generate noise from uniform distribution
        noise = (
            torch.rand(batch_size, noise_dimension).to(device) * 2 - 1
        )

        # Train discriminator
        # Real images
        real_images = data[0].to(device)
        disc_real_output = discriminator(real_images).flatten()
        disc_real_labels = torch.ones_like(disc_real_output)
        disc_real_loss = loss(disc_real_output, disc_real_labels)

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