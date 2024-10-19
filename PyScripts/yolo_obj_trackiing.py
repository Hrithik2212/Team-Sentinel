import cv2
import json
from tqdm import tqdm  # Import tqdm for progress bar
from ultralytics import YOLO, checks
from collections import defaultdict

checks()
model = YOLO("../Notebooks/Models/yolov11m_50epv2.pt")

def main(input_video_path = '../Data/Video/v3.mp4',     
         output_video_path = '../Results/object_detection_trackingv1.avi',  # Output video path with .avi extension
         needs_json_file = False,
         output_json_path = '../Results/object_detection_trackingv1.json'  # Output JSON path
        ):

    # Open input video
    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames in the video
    desired_fps = 15  # Limit FPS to 15

    # Calculate the frame interval to match the desired FPS
    frame_interval = int(input_fps / desired_fps)

    # Define video writer to save the processed video at 15 FPS in AVI format
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Use MJPG codec for better compatibility
    out = cv2.VideoWriter(output_video_path, fourcc, desired_fps, (width, height))

    # Initialize object counts and tracking variables
    object_counts = defaultdict(int)  # Dictionary to store counts for each object class
    frame_number = 0

    # Create a tqdm progress bar based on the total number of frames
    with tqdm(total=frame_count, desc="Processing Video", unit="frame") as pbar:
        # Process video frame by frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Only process frames that align with the desired FPS
            if frame_number % frame_interval == 0:
                # YOLO object detection and tracking
                results = model.track(frame, conf=0.6, verbose=False, persist=True)  # Use track method for object tracking

                # Draw bounding boxes, class names, confidence, and object IDs on the frame
                for result in results[0].boxes:
                    box = result.xyxy[0]  # Bounding box coordinates
                    class_id = int(result.cls[0])  # Class ID
                    class_name = model.names[class_id]  # Class name
                    conf = result.conf[0].item()  # Confidence score
                    object_id = result.id  # Tracking ID
                    object_id = str(int(object_id[0]))
                    # Update the object count for the class
                    if class_name not in object_counts:
                        object_counts[class_name] = set()

                    # Add the object ID to the set for this class
                    object_counts[class_name].add(object_id)

                    # Draw bounding box and information
                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
                    label = f'{class_name} {conf:.2f} ID: {object_id}'
                    cv2.putText(frame, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Write processed frame to output video
                out.write(frame)

            # Update the progress bar after processing each frame
            pbar.update(1)

            # Increment the frame counter
            frame_number += 1

    # Release resources
    cap.release()
    out.release()

    for object_class in object_counts.keys():
        object_counts[object_class] = sorted(list(object_counts[object_class]))
    # Save object counts to a JSON file
    
    if needs_json_file:
        with open(output_json_path, 'w') as f:
            json.dump(object_counts, f , indent=4)
            print(f"Object counts saved to {output_json_path}")

    print(f"Processed video saved to {output_video_path}")

if __name__ == "__main__":
    main(needs_json_file=True)
