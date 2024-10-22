
import cv2 
from typing import defaultdict
import base64
import json
from tqdm import tqdm
from collections import defaultdict
from deep_sort_realtime.deepsort_tracker import DeepSort
from server.controllers import utils , entity_extraction



def process_video(
    model , 
    classifier_model ,
    input_video_path,
    output_video_path,
    output_json_path,
    batch_size=48,  # Number of frames per batch
    blur_threshold=100.0  # Threshold for blur detection
):
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

                            # Detect blur and get Laplacian variance
                            is_blur, lap_var = utils.is_blurry(cropped_image, threshold=blur_threshold)

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
                            if 'eris' in utils.normalize_string(class_name):
                                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                                label = f'{class_name} ID: {track_id}'
                                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                            (0, 0, 255), 2)


                            else :
                                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                                label = f'{class_name} ID: {track_id}'
                                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                            (255, 0, 0), 2)

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

    # Prepare the final JSON output
    final_output = []

    # Process the best cropped images and analyze them
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
            # Encode image to base64 URL
            _, buffer = cv2.imencode('.png', cropped_image)
            encoded_image = base64.b64encode(buffer).decode('utf-8')
            mime_type = 'image/png'
            base64_image_url = f"data:{mime_type};base64,{encoded_image}"

            # Analyze the image using the Groq client (only once per track_id)

            if 'eris' in utils.normalize_string(class_name):
                print("fruit")
                entities = entity_extraction.perishable_analyze(base64_image_url , classifier_model)
            else :
                entities = entity_extraction.analyze_image(base64_image_url)


            # Append the result to the final output
            final_output.append({
                'track_id': track_id,
                'class_name': class_name,
                'image_base64_url': base64_image_url,
                'entities': entities
            })

    # Save the final output to a JSON file
    with open(output_json_path, 'w') as f:
        json.dump(final_output, f, indent=4)
        print(f"Final output saved to {output_json_path}")

    utils.convert_avi_to_mp4(output_video_path , output_video_path.replace('.avi' , '.mp4'))
    print(f"Processed video saved to {output_video_path.replace('.avi' , '.mp4')}")
