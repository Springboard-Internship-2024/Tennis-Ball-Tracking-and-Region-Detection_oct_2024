import streamlit as st
import torch
import cv2
import tempfile
import numpy as np
import os
import time

# Paths
repo_path = '.'
model_path = 'best.pt'

# Load the YOLOv5 model
model = torch.hub.load(repo_path, 'custom', path=model_path, source='local')

# Streamlit app UI
st.title('Tennis Player Detection App')
st.write('Upload a tennis video to detect players in real-time.')

# File uploader for video input
uploaded_video = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    # Preview the video
    st.subheader('Video Preview')
    st.video(uploaded_video)

    # Save the uploaded video to a temporary file
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_video_path = temp_video.name
    temp_video.write(uploaded_video.read())
    temp_video.close()

    # Open video capture
    cap = cv2.VideoCapture(temp_video_path)
    
    # Create a temporary file to save the processed video
    output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for video writing
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (640, 480))  # Adjust the frame size as needed

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection
        results = model(frame)
        frame = np.squeeze(results.render())  # Draw the detection boxes on the frame

        # Convert BGR to RGB for display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame in Streamlit
        stframe.image(frame, channels='RGB', use_column_width=True)

        # Write the processed frame to the output video
        out.write(frame)

        # Limit frame rate
        time.sleep(0.03)

    cap.release()
    out.release()

    st.success('Video processing complete!')

    # Cleanup
    os.unlink(temp_video_path)

    # Provide a download link for the processed video
    with open(output_video_path, 'rb') as file:
        st.download_button(label="Download Processed Video", data=file, file_name="processed_video.mp4")

    # Cleanup output video
    os.unlink(output_video_path)

