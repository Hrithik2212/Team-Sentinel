# app/routes/user.py

import time
from fastapi import APIRouter, Depends, HTTPException, status, Request, File, UploadFile
from sqlalchemy.orm import Session
from server.controllers.auth import verify_password, authenticate_request
from server.database.database import get_db
from server.database.schemas import User, UserCreate, Token, LoginRequest
from server.controllers import user_controller
from fastapi.security import OAuth2PasswordBearer
from typing import List
from fastapi.responses import JSONResponse
import base64

# Additional imports
import cv2
import json
from tqdm import tqdm
from ultralytics import YOLO, checks
from collections import defaultdict
import os
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from pathlib import Path
from groq import Groq
import tempfile
import subprocess



oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

router = APIRouter()

# Run checks and load the model at startup
checks()
model = YOLO("Notebooks/Models/yolov11m_50epv2.pt")

# Set the Groq API key (ensure it's stored securely)
os.environ["GROQ_API_KEY"] = "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"


@router.post("/users/", response_model=User)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    return user_controller.create_user(db=db, user=user)


@router.get("/getuser/", response_model=User)
@authenticate_request
async def get_current_user(request: Request, db: Session = Depends(get_db)):
    user_email = request.state.user.get("sub")
    user = user_controller.get_user(db, email=user_email)
    return user


@router.get("/user/all", response_model=List[User])
async def get_current_user(db: Session = Depends(get_db)):
    return user_controller.get_users(db)


@router.post("/token", response_model=Token)
def login_user(form_data: LoginRequest, db: Session = Depends(get_db)):

    db_user = user_controller.get_user(db, email=form_data.email)
    print(db_user)
    if db_user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    if not verify_password(form_data.password, db_user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user_controller.login(db_user)


# Utility functions
def is_blurry(image, threshold=100.0):
    """Determine if an image is blurry using the variance of the Laplacian."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold, laplacian_var


def analyze_image(base64_image_url):
    """
    Analyze the image using the Groq client to extract product entities.

    Args:
        base64_image_url (str): Base64 URL string of the image

    Returns:
        dict: Extracted entities from the image
    """
    # Initialize the Groq client
    client = Groq()

    # Define the prompt
    prompt = """
    Analyze the image of a grocery product and extract the following information: 
    - Brand name
    - Brand details (e.g., logo/tagline)
    - Pack size
    - Expiry date
    - MRP (Maximum Retail Price)
    - Product name
    - Count/quantity of items - count of the product present in the image 
    - Category of the product (e.g., personal care, household items, health supplements, etc.)

    Image description: The image is a low-resolution and blurry photo of a grocery product cropped from a conveyor belt.

    Image quality: The image is expected to have varying levels of blur and low resolution, with potential obstacles such as glare, shadows, or other environmental factors.

    Output format: JSON

    Handling edge cases:
    - If the image is blur, try to identify at least the brand name, do not leave the brand name N/A as it leads to unpleasant situations.
    - If the text is missing, please include 'N/A' in the corresponding field.
    - If no visible text or relevant information is present in the image, please return an empty JSON object or a message indicating 'N/A'.
    """

    # Prepare the API request
    completion = client.chat.completions.create(
        model="llama-3.2-90b-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": base64_image_url
                        }
                    }
                ]
            }
        ],
        temperature=0,
        max_tokens=1024,
        top_p=1,
        stream=False,
        stop=None,
        response_format={"type": "json_object"}  # Ensure response is in JSON format
    )

    # Parse the structured JSON response
    return json.loads(completion.choices[0].message.content)


@router.post("/upload-video/")
async def upload_video(video: UploadFile = File(...)):
    # Read the uploaded video file into bytes
    video_bytes = await video.read()

    # Write the video bytes to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
        temp_video_file.write(video_bytes)
        temp_video_path = temp_video_file.name

    # Prepare output paths
    with tempfile.TemporaryDirectory() as temp_dir:
        output_video_path = os.path.join(temp_dir, "processed_video.avi")
        output_json_path = os.path.join(temp_dir, "object_entities.json")

        # Call the main processing function
        process_video(
            input_video_path=temp_video_path,
            output_video_path=output_video_path,
            output_json_path=output_json_path
        )

        # Read the output JSON file
        with open(output_json_path, 'r') as f:
            final_output = json.load(f)

        # Read the processed video
        with open(output_video_path.replace('.avi' , '.mp4'), 'rb') as f:
            processed_video_bytes = f.read()

    # Encode the processed video in base64
    encoded_video = base64.b64encode(processed_video_bytes).decode('utf-8')

    # Clean up temporary video file
    os.remove(temp_video_path)

    # Return the JSON output and the processed video
    with open("Results/Sample_Response.json" , "w") as f:
        json.dump({"video": encoded_video, "products": final_output} , f , indent=4)
    return JSONResponse(content={"video": encoded_video, "products": final_output})


def process_video(
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
            entities = analyze_image(base64_image_url)

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

    convert_avi_to_mp4(output_video_path , output_video_path.replace('.avi' , '.mp4'))
    print(f"Processed video saved to {output_video_path.replace('.avi' , '.mp4')}")

def convert_avi_to_mp4(input_file, output_file):
    """
    Convert AVI to MP4 using FFmpeg with force overwrite.
    
    Args:
        input_file (str): Path to input AVI file
        output_file (str): Path for output MP4 file
    
    Returns:
        bool: True if successful, False if failed
    """
    try:
        command = ['ffmpeg', '-y', '-i', input_file, output_file]
        
        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        return process.returncode == 0
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return False