import streamlit as st
import cv2
import torch
import tempfile
import os
import numpy as np
import pathlib
import platform

# Use the appropriate Path class based on the operating system
if platform.system() == "Windows":
    PathClass = pathlib.WindowsPath
else:
    PathClass = pathlib.PosixPath




# Paths
repo_path = '.'
player_model_path = 'best.pt'  # Path to player detection model
ball_model_path = 'ball_best.pt'     # Path to ball detection model

# Load the YOLOv5 models
player_model = torch.hub.load(repo_path, 'custom', path=player_model_path, source='local')
ball_model = torch.hub.load(repo_path, 'custom', path=ball_model_path, source='local')

# Streamlit app UI
st.title('Tennis Detection App')
st.write('Upload a tennis video to detect players and balls in real-time.')

# File uploader for video input
uploaded_video = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    # Save the uploaded video to a temporary file
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.avi')  # Save as AVI
    temp_video_path = temp_video.name
    temp_video.write(uploaded_video.read())
    temp_video.close()

    # Output video file
    processed_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.avi').name

    # Open video capture
    cap = cv2.VideoCapture(temp_video_path)
    stframe = st.empty()

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Motion JPEG codec
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20  # Get FPS from input video or set default
    out = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Initialize the video writer once frame size is known
        if out is None:
            height, width, _ = frame.shape
            out = cv2.VideoWriter(processed_video_path, fourcc, fps, (width, height))

        # Perform player detection
        player_results = player_model(frame)
        player_boxes = player_results.xyxy[0].cpu().numpy()  # Get player bounding boxes

        # Perform ball detection
        ball_results = ball_model(frame)
        ball_boxes = ball_results.xyxy[0].cpu().numpy()  # Get ball bounding boxes

        # Annotate the frame
        annotated_frame = frame.copy()

        # Draw player detections (Blue)
        for box in player_boxes:
            x1, y1, x2, y2, conf, cls = map(int, box[:6])
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(annotated_frame, f'Player {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Draw ball detections (Green)
        for box in ball_boxes:
            x1, y1, x2, y2, conf, cls = map(int, box[:6])
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f'Ball {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write the processed frame to the output video (ensure it is in BGR)
        out.write(annotated_frame)

        # Convert BGR to RGB for display in Streamlit
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        stframe.image(annotated_frame, channels='RGB', use_container_width=True)

    cap.release()
    if out:
        out.release()

    st.success('Video processing complete!')

    # Provide download option for the processed video
    with open(processed_video_path, "rb") as video_file:
        st.download_button(
            label="Download Processed Video",
            data=video_file,
            file_name="processed_video.avi",  # Use .avi extension
            mime="video/avi"
        )

    # Cleanup temporary files
    os.unlink(temp_video_path)
    os.unlink(processed_video_path)

st.write("Ensure 'player_best.pt' and 'ball_best.pt' are in the same directory or provide the correct paths.")
