import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import sys

sys.setrecursionlimit(1500)



# Define function to load model from your GitHub repository
def load_model():
    device = torch.device('cpu')
    try:
        model = torch.hub.load('Omoze5/Facial-Expression-Detection', 'create_model', source='github', trust_repo=True)
        model.to(device)
        model.eval()
        print("Model loaded successfully on CPU.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model = load_model()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Function to make predictions
def predict(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Sidebar
st.sidebar.title('Files on GitHub')
proj_button = st.sidebar.button('Go to GitHub Projects')
github_url = 'https://github.com/Omoze5/Facial-Expression-Detection'
if proj_button:
    st.markdown(f'<a href="{github_url}" target="_blank">Click here to view the projects on GitHub</a>', unsafe_allow_html=True)

# Title of the app
st.title('Facial Expression Detection App')

# Project overview
st.text("""
Facial expressions are configurations of different micromotor (small muscle) 
movements in the face that are used to infer a person's discrete emotional 
state (e.g., happy, anger, neutral, fear, etc.).The aim of this project is 
to detect the facial expression of individuals. Facial expression detection 
systems utilize computer vision and machine learning techniques to analyze 
human facial expressions from images or video frames. These systems aim to 
identify and interpret facial expressions to understand human emotions 
and intentions.""")

option = st.selectbox('Upload a picture or take a new one?', ('Upload', 'Camera'))

# Handle image upload
if option == 'Upload':
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        Expression_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        if st.button('Predict'):
            if model:
                try:
                    prediction = predict(image)
                    label = Expression_labels[prediction]
                    st.write(f'Predicted Label: {label}')
                except Exception as e:
                    st.write(f"Error making prediction: {e}")
            else:
                st.write("Model not loaded successfully.")

# Handle camera capture
elif option == 'Camera':
    class VideoProcessor(VideoTransformerBase):
        def __init__(self):
            self.frame = None

        def recv(self, frame):
            self.frame = frame.to_image()
            return av.VideoFrame.from_image(self.frame)

    webrtc_ctx = webrtc_streamer(key="camera", video_processor_factory=VideoProcessor)
    if webrtc_ctx.video_processor:
        if st.button('Capture'):
            if webrtc_ctx.video_processor.frame is not None:
                img_pil = webrtc_ctx.video_processor.frame
                Expression_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
                if model:
                    try:
                        prediction = predict(img_pil)
                        label = Expression_labels[prediction]
                        st.write(f'Predicted Label: {label}')
                    except Exception as e:
                        st.write(f"Error making prediction: {e}")
                else:
                    st.write("Model not loaded successfully.")
            else:
                st.write("No frame captured.")
