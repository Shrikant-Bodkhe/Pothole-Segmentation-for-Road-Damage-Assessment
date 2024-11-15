import streamlit as st
import cv2
import time
import tempfile
from ultralytics import YOLO
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import moviepy.editor as mp

st.set_page_config(page_title="🚧 Pothole Detection System", layout="centered")
st.title("🚧 Pothole Detection System")

# Sidebar for uploading files and additional information
st.sidebar.title("📁 File Uploads")
st.sidebar.write("Please upload the YOLO model weights and the video file to start processing.")

# File uploaders in the sidebar
model_file = st.sidebar.file_uploader("Upload YOLO Model Weights (.pt)", type=["pt"])
video_file = st.sidebar.file_uploader("Upload a Video File", type=["mp4", "avi"])

# Check if both model and video files are uploaded
if model_file and video_file:
    # Save the model weights to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as temp_model_file:
        temp_model_file.write(model_file.read())
        model_path = temp_model_file.name

    # Load the YOLO model
    model = YOLO(model_path)

    # Save the uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
        temp_video_file.write(video_file.read())
        video_path = temp_video_file.name

    # Placeholder elements for dynamic updates
    status_text = st.empty()
    progress_bar = st.progress(0)
    col1, col2 = st.columns(2)
    eta_placeholder = col1.metric("⏳ ETA", "Calculating...")
    potholes_placeholder = col2.metric("🕳️ Unique Potholes Detected", "0")

    def cluster_potholes(centroids):
        """Cluster centroids to count unique potholes using DBSCAN."""
        clustering = DBSCAN(eps=50, min_samples=1).fit(centroids)
        return len(set(clustering.labels_)), clustering.labels_

    # Function to process video and count unique potholes dynamically
    def process_video(video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Error opening video file!")
            return

        # Video output settings
        output_path_avi = tempfile.NamedTemporaryFile(delete=False, suffix=".avi").name
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path_avi, fourcc, 20.0, (frame_width, frame_height))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processed_frames = 0
        start_time = time.time()

        all_centroids = []
        pothole_data = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run the model on each frame
            results = model(frame, conf=0.6)  # Higher confidence threshold
            boxes = results[0].boxes.xyxy.cpu().numpy()  # Get bounding boxes

            # Convert bounding boxes to centroids
            centroids = [
                ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) for box in boxes
            ]
            all_centroids.extend(centroids)

            # Draw detections on the frame
            annotated_frame = results[0].plot()
            out.write(annotated_frame)

            # Cluster centroids dynamically
            unique_pothole_count, _ = cluster_potholes(np.array(all_centroids))

            # Save frame data
            pothole_data.append({"Frame": processed_frames + 1, "Potholes Detected": len(centroids)})

            # Update progress and ETA
            processed_frames += 1
            elapsed_time = time.time() - start_time
            eta = elapsed_time / processed_frames * (total_frames - processed_frames)

            # Update dynamic elements
            progress_bar.progress(processed_frames / total_frames)
            eta_placeholder.metric("⏳ ETA", f"{int(eta // 60)}m {int(eta % 60)}s")
            potholes_placeholder.metric("🕳️ Unique Potholes Detected", unique_pothole_count)
            status_text.text(f"Processing frame {processed_frames}/{total_frames}")

        cap.release()
        out.release()

        # Convert AVI to MP4 for compatibility
        output_path_mp4 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        clip = mp.VideoFileClip(output_path_avi)
        clip.write_videofile(output_path_mp4, codec="libx264")

        # Create CSV file
        cluster_data = pd.DataFrame({
            "X": [c[0] for c in all_centroids],
            "Y": [c[1] for c in all_centroids],
        })
        csv_path = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name
        cluster_data.to_csv(csv_path, index=False)

        return unique_pothole_count, output_path_mp4, csv_path

    # Run video processing and display results
    unique_pothole_count, output_video_path, output_csv_path = process_video(video_path)

    # Display results
    potholes_placeholder.metric("🕳️ Unique Potholes Detected", unique_pothole_count)
    st.write(f"**Total Unique Potholes Detected: {unique_pothole_count}**")

    # Display and download processed video
    st.video(output_video_path)
    with open(output_video_path, "rb") as output_file:
        st.download_button(
            label="⬇️ Download Processed Video",
            data=output_file,
            file_name="processed_video.mp4",
            mime="video/mp4"
        )

    # Download CSV file
    with open(output_csv_path, "rb") as csv_file:
        st.download_button(
            label="⬇️ Download Pothole Data (CSV)",
            data=csv_file,
            file_name="pothole_data.csv",
            mime="text/csv"
        )

else:
    st.info("Please upload both the model weights and a video file to start processing.")
