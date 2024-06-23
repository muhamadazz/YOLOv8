import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os

def detect_objects_in_image(image_path, model_path="best1.pt"):
    model = YOLO(model_path)
    image = cv2.imread(image_path)
    if image is None:
        st.error("Image not found or unable to read.")
        return
    result = model.predict(image, show=False)
    result_image = result[0].plot()
    return result_image

def detect_objects_in_video(video_path, model_path="best1.pt"):
    model = YOLO(model_path)
    cam = cv2.VideoCapture(video_path)
    if not cam.isOpened():
        st.error("Error opening video stream or file.")
        return

    frames = []
    while True:
        ret, image = cam.read()
        if not ret:
            break
        result = model.predict(image, show=False)
        result_image = result[0].plot()
        frames.append(result_image)
    cam.release()
    return frames

st.title("Object Detection using YOLOv8")

choice = st.radio("Choose input type:", ('Image', 'Video'))

model_path = "best1.pt"

if choice == 'Image':
    image_file = st.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])
    if image_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(image_file.read())
            temp_path = temp_file.name
        result_image = detect_objects_in_image(temp_path, model_path)
        if result_image is not None:
            st.image(result_image, caption='Detected Objects', use_column_width=True)
        os.remove(temp_path)

elif choice == 'Video':
    video_file = st.file_uploader("Upload a Video", type=['mp4', 'avi', 'mov'])
    if video_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(video_file.read())
            temp_path = temp_file.name
        frames = detect_objects_in_video(temp_path, model_path)
        if frames:
            st.write("Detected Objects in Video:")
            for frame in frames:
                st.image(frame, use_column_width=True)
        os.remove(temp_path)
