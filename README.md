## LIVE AT : https://pothole-segmentation-for-road-damage-assessment-8d2uij6nqtexsh.streamlit.app

# Pothole Segmentation for Road Damage Assessment

This project aims to develop a pothole detection system using computer vision techniques, specifically leveraging the YOLO (You Only Look Once) model for segmentation. The goal is to identify potholes in dashcam footage and assess the road damage efficiently, which can aid in road maintenance and safety.

## Dataset Information

### Pothole Image Segmentation Dataset

The **Pothole Image Segmentation Dataset** is a computer vision dataset aimed at detecting and segmenting potholes for road safety and maintenance. It contains annotated images with high-resolution data for developing robust models to detect potholes in various scenarios.

#### Dataset Details:
- **Source**: This dataset was sourced from [Roboflow.com](https://www.kaggle.com/datasets/farzadnekouei/pothole-image-segmentation-dataset/data).
- **Number of Images**: 780 images
- **Format**: YOLOv8 annotations

#### Pre-processing Applied:
- Auto-orientation of pixel data (with EXIF-orientation stripping)
- Resized to 640x640 pixels

#### Augmentation Techniques (applied exclusively to the training data):
- 50% probability of horizontal flip
- Random cropping (0 to 20% of the image)
- Random rotation between -15 and +15 degrees
- Random shearing (up to 5° horizontally and vertically)
- Random brightness and exposure adjustment (up to ±25%)

#### Dataset Split:
- **Training Set**: 720 images
- **Validation Set**: 60 images

## Project Overview

This project focuses on pothole detection in videos captured by dashcams. By employing deep learning algorithms like YOLO for object detection and segmentation, the system identifies potholes and provides insights for road damage assessment. The model is designed to run offline using Python and Spyder IDE.

### Key Features:
- Detection of potholes from dashcam video footage
- Segmentation-based detection using YOLOv4
- Fully functional offline implementation using Python
- Simple GUI for user interaction

## Technologies Used

- **Python**: The core programming language used for development.
- **YOLOv4**: Object detection model used for segmentation of potholes.
- **OpenCV**: Library for video and image processing.
- **Tkinter**: Python library used to create the graphical user interface (GUI).
- **TensorFlow/Keras**: For training and running the YOLOv4 model.
- **Spyder**: Integrated Development Environment (IDE) used for offline implementation.

## Installation Instructions

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Shrikant-Bodkhe/Pothole-Segmentation-for-Road-Damage-Assessment.git
    cd Pothole-Segmentation-for-Road-Damage-Assessment
    ```

2. **Create a virtual environment (optional but recommended)**:
    ```bash
    python -m venv pothole-env
    source pothole-env/bin/activate  # On Windows use `pothole-env\Scriptsctivate`
    ```

3. **Install required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Run the GUI application**:
    After setting up the environment, you can run the GUI to interact with the pothole detection system.
    ```bash
    python app.py
    ```

2. **Upload a dashcam video**:
    The application will process the video, detect potholes, and display the segmented output.

3. **View Results**:
    The GUI will display the identified potholes on the video, and you can view the damage assessment in real-time.

## Model Training
From repo use *pothole_segmentation_for_road_damage_assessment.ipynb* for training

1. **Prepare the dataset**: 
    - Gather dashcam footage with labeled potholes.
    - Annotate the potholes using a tool like LabelImg.

2. **Train the YOLOv4 model**:
    - Follow the instructions in the [YOLOv4 repository](https://github.com/AlexeyAB/darknet) to train the model on your labeled data.

3. **Fine-tune the model**:
    - Use pre-trained weights and fine-tune the model for better accuracy on pothole detection.

4. **Save the model**:
    After training, save the model weights and configurations in the `runs/` folder.

## Contributing
Shrikant Bhaginath Bodkhe

