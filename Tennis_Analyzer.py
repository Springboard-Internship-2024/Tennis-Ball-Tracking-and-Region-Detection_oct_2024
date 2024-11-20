import os
import torch
import torch.nn as nn
import gdown
import streamlit as st
import tempfile
import cv2
from ultralytics import YOLO


# Define Keypoints Model
class KeypointsModel(nn.Module):
    def __init__(self):
        super(KeypointsModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 28)  # Adjusted to resolve size mismatch

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# Download model files
def download_file(url, file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(file_path):
        try:
            gdown.download(url, file_path, quiet=False)
        except Exception as e:
            st.error(f"Error downloading {file_path}: {e}")


# Load YOLO and Keypoints models
@st.cache_resource
def load_models():
    yolo_model_path = "models/best.pt"
    yolo_model_url = "https://drive.google.com/uc?export=download&id=1JVuj-ePLUiIm93wW-VYyRpgtwoeb7Lqr"
    keypoints_model_path = "models/keypoints_model.pth"
    keypoints_model_url = "https://drive.google.com/uc?export=download&id=1X6FAbjNLLlAhSEJTU3Xt7oVME7Tx_YbU"

    download_file(yolo_model_url, yolo_model_path)
    download_file(keypoints_model_url, keypoints_model_path)

    try:
        yolo_model = YOLO(yolo_model_path)
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None, None

    keypoints_model = KeypointsModel()
    try:
        state_dict = torch.load(keypoints_model_path, map_location="cpu")
        keypoints_model.load_state_dict(state_dict, strict=False)
        keypoints_model.eval()
    except Exception as e:
        st.error(f"Error loading Keypoints model: {e}")
        return yolo_model, None

    return yolo_model, keypoints_model


# Process video
def process_video(input_path, output_path, preview=False):
    yolo_model, keypoints_model = load_models()
    if yolo_model is None:
        st.error("YOLO model failed to load.")
        return False

    video = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.sidebar.progress(0)

    for i in range(frame_count):
        ret, frame = video.read()
        if not ret:
            break

        # YOLO Inference
        yolo_results = yolo_model(frame)
        annotated_frame = yolo_results[0].plot()

        # Keypoints Inference
        if keypoints_model:
            frame_tensor = torch.tensor(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            keypoints_results = keypoints_model(frame_tensor)
            keypoints = keypoints_results.detach().numpy().reshape(-1, 2)
            for (x, y) in keypoints:
                cv2.circle(annotated_frame, (int(x), int(y)), 5, (0, 255, 0), -1)

        # Write annotated frame
        out.write(annotated_frame)

        # Optional preview
        if preview:
            st.image(annotated_frame, channels="BGR", use_column_width=True)

        progress_bar.progress((i + 1) / frame_count)

    video.release()
    out.release()
    progress_bar.empty()
    return os.path.exists(output_path)


# Streamlit UI
st.title("🎾 Tennis Analyzer")
st.sidebar.title("Controls")
uploaded_file = st.sidebar.file_uploader("📂 Upload Video", type=["mp4", "avi", "mov"])
preview_option = st.sidebar.checkbox("Preview During Processing", value=False)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as temp_input:
        temp_input.write(uploaded_file.read())
        temp_input_path = temp_input.name

    temp_output_path = tempfile.mktemp(suffix=".mp4")

    if st.sidebar.button("Process Video"):
        st.sidebar.text("Processing...")
        success = process_video(temp_input_path, temp_output_path, preview=preview_option)

        if success:
            st.sidebar.text("Processing Complete!")
            st.video(temp_output_path)
            with open(temp_output_path, "rb") as file:
                st.sidebar.download_button("⬇️ Download Processed Video", data=file, file_name="processed_video.mp4")
        else:
            st.sidebar.error("Processing Failed.")

    # Ensure proper cleanup
    try:
        os.remove(temp_input_path)
    except Exception as e:
        st.error(f"Error removing temporary input file: {e}")

    if os.path.exists(temp_output_path):
        os.remove(temp_output_path)
else:
    st.info("Upload a video to process.")
