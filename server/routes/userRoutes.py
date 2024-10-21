# app/routes/user.py
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
import cv2
import json
from tqdm import tqdm
from ultralytics import YOLO, checks
from collections import defaultdict
import os
from deep_sort_realtime.deepsort_tracker import DeepSort
from groq import Groq
import tempfile
import subprocess
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import requests
import io
import unicodedata

def normalize_string(s):
    """
    Normalize Unicode characters, remove leading/trailing whitespace,
    and convert to lowercase.
    """
    return unicodedata.normalize('NFKC', s).strip().lower()



oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

router = APIRouter()

 
def get_classifier_model():
    model = models.mobilenet_v2(pretrained=False)

    # Modify the classifier for binary classification
    num_classes = 2  # 'fresh' and 'rotten'
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    # Load the saved weights
    checkpoint_path = "Notebooks/Models/mobilenetv2_freshness_classifier.pth"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # Move the model to the device
    model = model.to(device)

    # Set the model to evaluation mode
    model.eval()
    return model

# Run checks and load the model at startup
checks()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO("Notebooks/Models/yolov11m_50epv3.pt")
os.environ["GROQ_API_KEY"] = "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"
classifier_model = get_classifier_model()




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

def classify_image(image_input, classifier_model, device, threshold=0.5):
    """
    Classify an image using the provided classifier_model and threshold.

    Parameters:
    - image_input (PIL.Image.Image): A PIL Image object.
    - classifier_model (torch.nn.Module): The pre-trained and loaded PyTorch classifier_model.
    - device (torch.device): The device to run the classifier_model on (CPU or GPU).
    - threshold (float): Threshold for classifying as 'rotten'. Default is 0.5.

    Returns:
    - predicted_class (str): The predicted class label ('fresh' or 'rotten').
    - probability (float): The probability of the image belonging to the 'rotten' class.
    """

    # Define the transformation pipeline
    preprocess = transforms.Compose([
        transforms.Resize(256),                # Resize the shorter side to 256 pixels
        transforms.CenterCrop(224),            # Crop the center 224x224 pixels
        transforms.ToTensor(),                 # Convert PIL Image to Tensor
        transforms.Normalize(                  # Normalize with ImageNet mean and std
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Ensure the input is a PIL Image
    if isinstance(image_input, Image.Image):
        image = image_input.convert('RGB')
    else:
        raise ValueError("image_input must be a PIL Image.")

    # Apply the preprocessing transformations
    input_tensor = preprocess(image)

    # Create a mini-batch as expected by the classifier_model
    input_batch = input_tensor.unsqueeze(0).to(device)  # Shape: [1, 3, 224, 224]

    with torch.no_grad():
        # Get classifier_model outputs
        outputs = classifier_model(input_batch)  # Shape: [1, num_classes]

        # Apply softmax to get probabilities
        probabilities = nn.functional.softmax(outputs, dim=1)

        # Get the probability of the positive class (class 1)
        prob_positive = probabilities[0][1].item()

        # Determine the predicted class based on the threshold
        if prob_positive >= threshold:
            predicted_class = 'rotten'  # Assuming class 1 is 'rotten'
        else:
            predicted_class = 'fresh'   # Assuming class 0 is 'fresh'

    return predicted_class, prob_positive

def image_to_base64_url_from_image(image):
    """
    Convert a PIL Image object to a base64 URL string.
    
    Args:
        image (PIL.Image.Image): PIL Image object
        
    Returns:
        str: Base64 URL string of the image
    """
    try:
        # Save image to bytes buffer
        buffered = io.BytesIO()
        image.save(buffered, format='PNG')
        encoded_string = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Create the complete base64 URL
        base64_url = f"data:image/png;base64,{encoded_string}"

        return base64_url

    except Exception as e:
        raise Exception(f"Error converting image to base64: {str(e)}")

def analyze_freshness(image):
    """
    Analyze the image using the Groq API to extract product information.

    Parameters:
    - image (PIL.Image.Image): A PIL Image object.

    Returns:
    - dict: Dictionary containing the analysis results.
    """
    # Import the Groq client
    from groq import Groq  # Ensure you have the groq package installed

    # Ensure the API key is set
    if "GROQ_API_KEY" not in os.environ:
        raise Exception("Please set the GROQ_API_KEY environment variable.")

    # Initialize the Groq client
    client = Groq()
    fruit_prompt = """
Analyze the image of a grocery product and extract the following information:
- Name of the product (e.g., apple, banana, bread, etc.)
- Count/quantity of items - count of the product present in the image
- Category of the product (e.g., fruit, vegetable, bread)
- Estimated shelf life (in terms of days)

Image description: The image is a low-resolution and blurry photo of a grocery product cropped from a conveyor belt.

Image quality: The image is expected to have varying levels of blur and low resolution, with potential obstacles such as glare, shadows, or other environmental factors.

Output format: JSON

Handling edge cases:
- If the product is a fruit or vegetable, assess its freshness and estimate its shelf life based on visual cues.
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
                        "text": fruit_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_to_base64_url_from_image(image)
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
    try:
        response_content = completion.choices[0].message.content
        return json.loads(response_content)
    except json.JSONDecodeError:
        # Handle cases where the response is not valid JSON
        return {"error": "Invalid JSON response from API."}
    except Exception as e:
        # Handle any other exceptions
        return {"error": str(e)}

def perishable_analyze(image_url, threshold=0.9):
    """
    Process the image from the given URL and return the analysis results.

    Parameters:
    - image_url (str): URL of the image to process (HTTP/HTTPS or data URL).
    - threshold (float): Threshold for classifying as 'rotten'. Default is 0.7.

    Returns:
    - dict: Dictionary containing the analysis results.
    """
    # Step 1: Load the image
    try:
        if image_url.startswith('data:'):
            # It's a data URL; decode it
            header, encoded = image_url.split(',', 1)
            image_data = base64.b64decode(encoded)
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
        else:
            # It's an HTTP/HTTPS URL; download the image
            response = requests.get(image_url)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert('RGB')
    except Exception as e:
        return {"error": f"Failed to download or open image: {str(e)}"}

    # Step 2: Classify the image
    predicted_label, probability = classify_image(image, classifier_model, device, threshold=threshold)

    # Step 3: Analyze the image using Groq API
    analysis_result = analyze_freshness(image)
    if 'error' in analysis_result:
        return analysis_result  # Return the error

    # Step 4: Combine the results
    result = {
        "product_name": analysis_result.get("product_name", None),
        "count": analysis_result.get("count", None),
        "category": analysis_result.get("category", None),
        "estimated_shelf_life": analysis_result.get("estimated_shelf_life", None),
        "state": predicted_label,  # classifier_model prediction ('fresh' or 'rotten')
        "freshness": (1-probability)   # classifier_model probability
    }
    print(result)
    return result


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
                            if 'eris' in normalize_string(class_name):
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

            if 'eris' in normalize_string(class_name):
                print("fruit")
                entities = perishable_analyze(base64_image_url)
            else :
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

from pathlib import Path
    
def image_to_base64_url_bytes(image_bytes: bytes, filename: str) -> str:
    """
    Convert image bytes and filename to a base64 data URL string.
    
    Args:
        image_bytes (bytes): The byte content of the image.
        filename (str): The original filename of the image.
        
    Returns:
        str: The base64-encoded data URL of the image.
    
    Raises:
        ValueError: If the file extension is unsupported.
    """
    # Get the file extension
    extension = Path(filename).suffix.lower()
    
    # Map extensions to MIME types
    mime_type_mapping = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.svg': 'image/svg+xml'
    }
    
    mime_type = mime_type_mapping.get(extension)
    if not mime_type:
        raise ValueError(f"Unsupported file extension: {extension}")
    
    # Encode image bytes to base64
    try:
        encoded_string = base64.b64encode(image_bytes).decode('utf-8')
    except Exception as e:
        raise ValueError(f"Error encoding image to base64: {e}")
    
    # Create the data URL
    base64_url = f"data:{mime_type};base64,{encoded_string}"
    
    return base64_url

from typing import List

@router.post("/upload-image/", summary="Upload an image and receive its base64-encoded URL")
async def upload_image(image: List[UploadFile] = File(...)):
    try:
        image_bytes = await image[0].read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image file: {e}")
    
    # Convert to base64 URL
    try:
        base64_url = image_to_base64_url_bytes(image_bytes, image[0].filename)
        return JSONResponse(analyze_image(base64_image_url=base64_url))
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
    
