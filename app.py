import streamlit as st
import torch
import tempfile
import numpy as np
import pathlib
from pathlib import Path
import sys
import os

# Add the path to sys.path for OpenCV
opencv_path = os.path.join(os.path.dirname(__file__), 'opencv')
sys.path.append(os.path.join(opencv_path, 'cv2'))
import cv2

# Adjust pathlib for Windows if needed
model1_path = "tplayer_best.pt"
model2_path ="tball_best.pt"

# Load both models
model1 = torch.hub.load("ultralytics/yolov5", "custom", path=model1_path)
model2 = torch.hub.load("ultralytics/yolov5", "custom", path=model2_path)

st.markdown("""
    <style>
        .main {
            max-width: 900px;
            padding: 20px;
            background-color: #ffffff;
            border-radius: 15px;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
        }
        
        .css-18e3th9 {
            background: linear-gradient(135deg, #f6d365, #fda085);
        }
        
        h1 {
            color: #283593;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-weight: bold;
            text-shadow: 1px 1px 4px rgba(0,0,0,0.2);
        }
        
        .stButton>button {
            background-color: #0288d1;
            color: white;
            padding: 10px 20px;
            border-radius: 10px;
            border: none;
            cursor: pointer;
            font-weight: bold;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #0277bd;
        }
        
        .stDownloadButton > button {
            background-color: #388e3c;
            color: white;
            padding: 10px 20px;
            border-radius: 10px;
            font-weight: bold;
            font-size: 16px;
            border: none;
            cursor: pointer;
        }
        .stDownloadButton > button:hover {
            background-color: #2e7d32;
        }
    </style>
""", unsafe_allow_html=True)

# App title and description
st.title("🎾 Tennis Game Tracker")
st.write("Detect players and balls in a tennis match using YOLOv5. Upload a video, run detection, and download the processed result.")

col1, col2 = st.columns(2)

with col1:
    st.write("## Video Preview")

with col2:
    st.write("## Actions")

    video_file = st.file_uploader("Upload a video", type=["mp4", "mov"])

    if 'processed' not in st.session_state:
        st.session_state['processed'] = False
    if 'output_path' not in st.session_state:
        st.session_state['output_path'] = None
    if 'upload_path' not in st.session_state:
        st.session_state['upload_path'] = None
    if 'processing_msg' not in st.session_state:
        st.session_state['processing_msg'] = None

    if video_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_file.read())
            st.session_state['upload_path'] = tmp_file.name
        st.success("Video uploaded successfully!")
        st.session_state['processed'] = False
        st.session_state['output_path'] = None
        st.session_state['processing_msg'] = None

    if st.session_state['upload_path'] and st.button("Preview Uploaded Video"):
        with col1:
            st.video(st.session_state['upload_path'])

    if st.button("Run Model") and video_file is not None and not st.session_state['processed']:
        st.session_state['processing_msg'] = st.info("Processing video, please wait...")

        # Open the uploaded video
        video = cv2.VideoCapture(st.session_state['upload_path'])

        # Setup output file for saving processed video
        output_tempfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        st.session_state['output_path'] = output_tempfile.name
        output_tempfile.close()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = video.get(cv2.CAP_PROP_FPS)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(st.session_state['output_path'], fourcc, fps, (width, height))

        # Process frames with both models
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            
            # Apply the first model
            results1 = model1(frame)
            frame_after_model1 = np.squeeze(results1.render())
            
            # Apply the second model on the output of the first model
            results2 = model2(frame_after_model1)
            annotated_frame = np.squeeze(results2.render())
            
            out.write(annotated_frame)

        video.release()
        out.release()

        st.session_state['processed'] = True
        st.session_state['processing_msg'].empty()
        st.success("Processing complete!")

    if st.session_state['output_path'] and st.session_state['processed']:
        with col1:
            try:
                if Path(st.session_state['output_path']).exists():
                    st.video(st.session_state['output_path'])
                else:
                    st.error("Output video file not found.")
            except Exception as e:
                st.error(f"Error loading video: {e}")

    if st.session_state['output_path'] and st.session_state['processed']:
        with open(st.session_state['output_path'], "rb") as f:
            st.download_button(
                label="Download Processed Video",
                data=f,
                file_name="processed_video.mp4",
                mime="video/mp4"
            )
