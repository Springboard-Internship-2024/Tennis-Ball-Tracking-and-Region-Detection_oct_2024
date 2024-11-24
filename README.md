# Tennis Ball Detection using YOLOv5 and Streamlit

This project demonstrates real-time tennis ball and player detection from video input using a custom-trained YOLOv5 model. The application is built with Streamlit for an interactive user interface.

**[Live Application](https://tennisballandperson.streamlit.app)**

## Features

- Upload video files for processing
- Displays the progress of object detection
- Shows output video with tennis ball and player detection after processing

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd yolov5
```

### 2. Install Dependencies

Make sure you have Python installed. Then, install the required libraries:

```bash
pip install -r requirements.txt
```

### 3. Model Setup

Place the custom-trained YOLOv5 model file in the `yolov5/runs/exp/weights/best.pt`.

### 4. Run the Application Locally

Run the following command in the `yolov5` directory:

```bash
streamlit run app.py
```
## Features

- Upload video files for processing
- Displays the progress of object detection
- Shows output video with tennis ball and player detection after processing
