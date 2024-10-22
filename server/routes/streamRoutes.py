from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from server.controllers import utils, video_stream , freshness_classfier ,entity_extraction ## Core ML Logic Files
from typing import List
from fastapi.responses import JSONResponse
import base64
import json
from ultralytics import YOLO, checks
import os
import tempfile

# Run checks and load the model at startup
checks()
device = freshness_classfier.device
model = YOLO("Notebooks/Models/yolov11m_50epv3.pt")
os.environ["GROQ_API_KEY"] = "" ## PUT YOUR API KEY
classifier_model = freshness_classfier.get_classifier_model()

router = APIRouter()

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
        video_stream.process_video(
            model,
            classifier_model,
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



@router.post("/upload-image/", summary="Upload an image and receive its base64-encoded URL")
async def upload_image(image: List[UploadFile] = File(...)):
    try:
        image_bytes = await image[0].read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image file: {e}")
    try:
        base64_url = utils.image_to_base64_url_bytes(image_bytes, image[0].filename)
        return JSONResponse(entity_extraction.analyze_image(base64_image_url=base64_url))
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
    
