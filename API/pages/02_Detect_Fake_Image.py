import os
import sys
import streamlit as st
import pandas as pd
from io import StringIO
import torch
import torchvision.transforms as transforms
from PIL import Image

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, project_root)

from models.WGAN_GP import WGAN_GP
from models.DCGAN import DCGAN

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    uploaded_file = Image.open(uploaded_file)
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    transformed_image = transform(uploaded_file)
    transformed_image = transformed_image.unsqueeze(0)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    critic = DCGAN.Discriminator(
        first_conv_size=128, input_channels=3
    )

    checkpoint = torch.load(
        "parameters/DCGAN-10epochs/discriminator.pth", map_location=device
    )

    # Check if the checkpoint contains just the state_dict or is a full checkpoint
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        critic.load_state_dict(checkpoint["state_dict"])
    elif isinstance(checkpoint, dict) and "critic_state_dict" in checkpoint:
        critic.load_state_dict(checkpoint["critic_state_dict"])
    else:
        # If checkpoint is just the state_dict itself
        critic.load_state_dict(checkpoint)

    critic.to(device)
    critic.eval()

    probability = critic(transformed_image)
    # probability = torch.sigmoid(critic_output)

    # st.write(probability)
    st.write(f"Probability that the image is fake: {(100 - probability[0][0] * 100):.4f}%")
