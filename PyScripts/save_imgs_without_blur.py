import cv2
import json
from tqdm import tqdm
from ultralytics import YOLO, checks
from collections import defaultdict
import os
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

checks()
model = YOLO("../Notebooks/Models/yolov11m_50epv3.pt")

def is_blurry(image, threshold=100.0):
    """Determine if an image is blurry using the variance of the Laplacian."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold, laplacian_var

def main(
    input_video_path='../Data/Video/v4.mp4',
    output_video_path='../Results/v4.avi',
    needs_json_file=False,
    output_json_path='../Results/object_detection_trackingv1.json',
    batch_size=32,  # Number of frames per batch
    cropped_images_folder='../Results/Cropped_Images',  # Folder to save cropped images
    blur_threshold=100.0  # Threshold for blur detection
):
    # Create the cropped images folder if it doesn't exist
    os.makedirs(cropped_images_folder, exist_ok=True)

    # Open input video
    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    desired_fps = 15  # Limit FPS to 15

    # Calculate the frame interval to match the desired FPS
    frame_interval = int(input_fps / desired_fps) if input_fps > desired_fps else 1

    # Define video writer to save the processed video at desired FPS in AVI format
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, desired_fps, (width, height))

    # Initialize object counts and tracking variables
    object_counts = defaultdict(set)  # Use set to avoid duplicate IDs
    track_info = defaultdict(list)  # Dictionary to store candidates for each track_id
    frame_number = 0

    # Initialize batch processing variables
    batch_frames = []
    batch_frame_numbers = []

    # Initialize the DeepSort tracker
    tracker = DeepSort(max_age=5)

    # Progress bar
    with tqdm(total=frame_count, desc="Processing Video", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Only process frames that align with the desired FPS
            if frame_number % frame_interval == 0:
                batch_frames.append(frame)
                batch_frame_numbers.append(frame_number)

                # When batch is full or it's the last frame, process the batch
                if len(batch_frames) == batch_size or frame_number == frame_count - 1:
                    # Run detection on the batch of frames
                    results = model(batch_frames, conf=0.6, verbose=False)

                    # Process each frame individually for tracking
                    for idx, (res, frame) in enumerate(zip(results, batch_frames)):
                        # Prepare detections for DeepSort
                        detections = []
                        for box in res.boxes:
                            xyxy = box.xyxy[0].cpu().numpy()  # Bounding box coordinates
                            conf = box.conf[0].cpu().numpy()  # Confidence score
                            class_id = int(box.cls[0])
                            class_name = model.names[class_id]
                            # Detection format for DeepSort: [xmin, ymin, width, height]
                            x1, y1, x2, y2 = xyxy
                            w = x2 - x1
                            h = y2 - y1
                            detections.append(([x1, y1, w, h], conf, class_name))

                        # Update tracker with detections
                        tracks = tracker.update_tracks(detections, frame=frame)

                        # Process tracks
                        for track in tracks:
                            if not track.is_confirmed():
                                continue
                            track_id = track.track_id
                            ltrb = track.to_ltrb()
                            class_name = track.get_det_class()
                            x1, y1, x2, y2 = ltrb
                            # Update object counts
                            object_counts[class_name].add(track_id)

                            # Calculate bounding box area
                            bbox_area = (x2 - x1) * (y2 - y1)

                            # Ensure coordinates are within image boundaries
                            x1_crop = max(0, int(x1))
                            y1_crop = max(0, int(y1))
                            x2_crop = min(frame.shape[1], int(x2))
                            y2_crop = min(frame.shape[0], int(y2))

                            cropped_image = frame[y1_crop:y2_crop, x1_crop:x2_crop]

                            # Skip if cropped image is empty
                            if cropped_image.size == 0:
                                continue

                            # Detect blur and get laplacian variance
                            is_blur, lap_var = is_blurry(cropped_image, threshold=blur_threshold)

                            # Store candidate image info
                            track_info[track_id].append({
                                'area': bbox_area,
                                'lap_var': lap_var,
                                'is_blurry': is_blur,
                                'frame': frame.copy(),
                                'bbox': (x1, y1, x2, y2),
                                'class_name': class_name
                            })

                            # Draw bounding box and label on the frame
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                            label = f'{class_name} ID: {track_id}'
                            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                        # Write the processed frame to the output video
                        out.write(frame)

                        # Update progress bar
                        pbar.update(1)

                    # Clear the batch
                    batch_frames = []
                    batch_frame_numbers = []

            else:
                # Update the progress bar even if the frame is skipped
                pbar.update(1)

            frame_number += 1

    # Release resources
    cap.release()
    out.release()

    # Save the best cropped images for each track_id
    for track_id, candidates in track_info.items():
        # Sort candidates by area (largest first)
        candidates_sorted = sorted(candidates, key=lambda x: x['area'], reverse=True)

        # Try to find the first non-blurry image
        for candidate in candidates_sorted:
            if not candidate['is_blurry']:
                # Use this candidate
                selected_candidate = candidate
                break
        else:
            # All images are blurry; select the one with the highest Laplacian variance
            selected_candidate = max(candidates_sorted, key=lambda x: x['lap_var'])

        x1, y1, x2, y2 = selected_candidate['bbox']
        class_name = selected_candidate['class_name']
        frame = selected_candidate['frame']

        x1_crop = max(0, int(x1))
        y1_crop = max(0, int(y1))
        x2_crop = min(frame.shape[1], int(x2))
        y2_crop = min(frame.shape[0], int(y2))

        cropped_image = frame[y1_crop:y2_crop, x1_crop:x2_crop]

        if cropped_image.size > 0:
            # Save image with class name and track_id
            image_path = os.path.join(cropped_images_folder, f"{class_name}_{track_id}.png")
            cv2.imwrite(image_path, cropped_image)

    # Convert sets to sorted lists for JSON serialization
    for class_name in object_counts.keys():
        object_counts[class_name] = sorted(list(object_counts[class_name]))

    # Save object counts to a JSON file
    if needs_json_file:
        with open(output_json_path, 'w') as f:
            json.dump(object_counts, f, indent=4)
            print(f"Object counts saved to {output_json_path}")

    print(f"Processed video saved to {output_video_path}")
    print(f"Cropped images saved to {cropped_images_folder}")

if __name__ == "__main__":
    main(needs_json_file=True)
