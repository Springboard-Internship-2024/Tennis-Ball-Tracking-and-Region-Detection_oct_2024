Tennis Player and Ball Detection using YOLOv5 and Streamlit

Experience the power of AI with a cutting-edge application designed to detect tennis players and balls in real-time. This interactive tool uses a custom-trained YOLOv5 model and a seamless Streamlit interface to bring your tennis match analyses to life.

Try the Live App!


---

Features

Video Upload: Supports .mp4 video uploads for processing.

Real-Time Detection: Tracks tennis players and ball positions in real-time.

Progress Visualization: Stay updated with detection progress during video analysis.

Output Video Playback: View enhanced videos with highlighted detections post-processing.



---

Quickstart Guide

1. Clone the Repository

git clone <repository-url>
cd yolov5

2. Install Dependencies

Ensure you have Python installed (version 3.8 or higher). Then install the required libraries:

pip install -r requirements.txt

3. Prepare the Model

Place your trained YOLOv5 model weights (best.pt) in the specified directory:

yolov5/runs/exp/weights/best.pt

4. Run the Streamlit App

Navigate to the yolov5 directory and launch the application:

streamlit run app.py


---

Folder Structure

yolov5/
├── app.py                  # Streamlit application entry point
├── runs/
│   └── exp/
│       └── weights/
│           └── best.pt     # YOLOv5 model weights file
├── data/                   # Input video directory
└── outputs/                # Processed output videos


---

Workflow

1. Open the application locally or use the live app.


2. Upload a video (supported format: .mp4).


3. The app processes the video and displays a progress bar.


4. After completion, watch the enhanced video with tennis ball and player detections.




---

Dependencies

The following tools and libraries power this project:

Streamlit: For creating the user-friendly interface.

PyTorch: For utilizing the YOLOv5 model.

OpenCV: For video processing and visualization.

YOLOv5: The backbone object detection framework.



---

Example Use Case

Tennis Analytics: Analyze player movements, ball trajectories, and match dynamics effortlessly by uploading your game recordings.

Coaching Tools: Provide actionable insights to players by identifying patterns and tendencies.

Highlight Reel Creation: Enhance video footage with dynamic overlays for professional presentations.



---

Get started with tennis tracking today! 🚀
