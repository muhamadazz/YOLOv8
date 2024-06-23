from ultralytics import YOLO
import cv2
import time

def detect_objects_in_image(image_path, model_path="best1.pt"):
    # Load model
    model = YOLO(model_path)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to read.")
    
    # Predict
    result = model.predict(image, show=True)
    
    # Display the result
    cv2.imshow("Detection", result[0].plot())
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_objects_in_video(video_path, model_path="best1.pt"):
    # Load model
    model = YOLO(model_path)
    
    # Load video
    cam = cv2.VideoCapture(video_path)
    if not cam.isOpened():
        raise ValueError("Error opening video stream or file.")
    
    while True:
        ret, image = cam.read()
        if not ret:
            break
        
        _time_mulai = time.time()
        result = model.predict(image, show=False)
        
        # Display the result
        result_image = result[0].plot()
        cv2.imshow("Detection", result_image)
        print("Detection time:", time.time() - _time_mulai)
        
        _key = cv2.waitKey(1)
        if _key == ord('q'):
            break
    
    cam.release()
    cv2.destroyAllWindows()

def main():
    choice = input("Enter 'image' to detect objects in an image or 'video' to detect objects in a video: ").strip().lower()
    if choice == 'image':
        image_path = input("Enter the path to the image: ").strip()
        detect_objects_in_image(image_path)
    elif choice == 'video':
        video_path = input("Enter the path to the video: ").strip()
        detect_objects_in_video(video_path)
    else:
        print("Invalid choice. Please enter 'image' or 'video'.")

if __name__ == "__main__":
    main()
