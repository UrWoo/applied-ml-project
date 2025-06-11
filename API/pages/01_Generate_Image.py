from io import BytesIO
import requests
import streamlit as st
from PIL import Image
import uvicorn
import threading
import time
from wgan_api import app

def start_fastapi_server():
    uvicorn.run(app, host="127.0.0.1", port=8000)

if 'server_started' not in st.session_state:
    st.session_state.server_started = False

st.title("Image Display from FastAPI")

if not st.session_state.server_started:
    if st.button("🚀 Start WGAN Server"):
        # Start server in daemon thread
        server_thread = threading.Thread(target=start_fastapi_server, daemon=True)
        server_thread.start()
        st.session_state.server_started = True
        st.success("Server starting... Please wait a moment.")
        time.sleep(3)

# Main functionality
if st.session_state.server_started:
    ENDPOINT_URL = "http://127.0.0.1:8000/image"
    
    st.success("🟢 WGAN Server is running")
    
    if st.button("Get Image"):
        try:
            response = requests.get(ENDPOINT_URL)
            
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                st.image(image, caption="Image from WGAN-GP")
            else:
                st.error(f"Failed to fetch image. Status code: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to FastAPI server. Make sure it's running.")
        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.info("👆 Please start the WGAN server first")
