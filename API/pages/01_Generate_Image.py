from io import BytesIO
import requests
import streamlit as st
from PIL import Image


# API_URL = ""
ENDPOINT_URL = f"http://0.0.0.0:8000/image"

st.title("Image Display from FastAPI")

if st.button("Get Image"):
    try:
        response = requests.get(ENDPOINT_URL)
        
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            
            st.image(image, caption="Image from WGAN-GP")
        else:
            st.error("Failed to fetch image")
            
    except Exception as e:
        st.error(f"Error: {e}")