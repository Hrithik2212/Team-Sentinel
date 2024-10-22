# Smart-Vision-System-Ecommerce-Warehouse

# Demonstration 

* The goal is to immitate conveyor belt and to detect , track , classify , perform OCR and extract entities on ecommerce products moving on a conveyor belt 

[![Watch the video](https://img.youtube.com/vi/Z_sjBRmlVX8/0.jpg)](https://youtu.be/Z_sjBRmlVX8?t=2s)


# Smart Vision System Architecture

The **Smart Vision System** leverages AWS cloud infrastructure and machine learning models to provide real-time video processing, image classification, object detection, object tracking, OCR, and entity extraction. The system is designed to securely stream video from a conveyor belt, analyze the footage for key insights, and display the results to operators through an intuitive dashboard.

![image](https://github.com/user-attachments/assets/b7ffe4ad-b97f-46a7-84f9-843e932c54e8)


### 1.Input
- Video streams from an on-premises camera monitoring a conveyor belt are securely transmitted to the cloud via a VPN gateway.

### 2.Processing
- Backend services running on EC2 instances (FastAPI) handle video processing.
- Machine learning algorithms such as **YOLOv11**, **MobileNetV2**, **Llama3.2-11B-vision**, and **DeepSort** are used for object detection, classification, OCR, object tracking, and entity extraction.

### 3.Storage
- Processed images are stored in **S3 buckets**.
- Structured data such as object tracking logs and entities are stored in **DynamoDB**.

### 4.Monitoring
- **AWS CloudWatch** is used to log system metrics and ensure smooth operation.

### 5.Output
- Operators interact with the processed data and video streams through a **ReactJS frontend** hosted on **AWS Amplify**.


# Technology Stack

### Cloud Platform:
- **AWS VPC**, **EC2 instances**, **S3 buckets**, **DynamoDB**, **AWS Bedrock**, **AWS Sagemaker**, **AWS Amplify**

### Backend Framework:
- **FastAPI** for processing video and managing backend services (Server-Sent Events for real-time processing).

### Frontend Framework:
- **ReactJS** for the operator dashboard, hosted using **AWS Amplify**.

## Hardware Specifications:
- **Camera**: High-definition camera mounted on the conveyor belt to capture real-time footage.
- **Compute Power**: EC2 instance and **AWS Sagemaker GPU support** (e.g., p3.2xlarge) for real-time video processing.

## Storage:
- **S3 Buckets** for image storage and long-term archival.
- **DynamoDB** for storing logs and processed data entries.

## Machine Learning Models:
- **YOLOv11**: Used for object detection to identify items on the conveyor belt.
- **MobileNetV2**: Used for classification tasks, such as determining object freshness or quality.
- **DeepSort**: Used for object tracking to precisely track the objects on the conveyor belt without duplicate entries.
- **Llama3.2-11b-vision**: Used for OCR and entity extraction from the detected objects.

## Training Data:

### Dataset:
- Imitated conveyor belt and manually collected data for object detection.
- Used the **Kaggle Fresh Fruits and Vegetables Dataset** for classification:
  - [Roboflow Conveyor Belt Dataset](https://app.roboflow.com/flipkart-grid-gqtdk/belt-nublw/2)
  - [Kaggle Fresh and Stale Classification Dataset](https://www.kaggle.com/datasets/swoyam2609/fresh-and-stale-classification)

### Annotations:
- Manually annotated data for object detection using **Roboflow Annotator**.

### Training Tools:
- **Kaggle P100 GPU** was used for training and fine-tuning the models used in the system.
- **Transfer Learning**: Pre-trained models (e.g., **MobileNetV2** and **YOLOv11**) were fine-tuned with the custom dataset.

## Simulation Video:
- A simulation of the system demonstrates how the camera captures real-time footage, how the backend processes the video, and how the operator dashboard displays the results.


## File Structure (Nvigate throught the project)


```
├── client/                   # React frontend application
│   ├── favicon.svg                   # Website favicon
│   ├── index.html                    # Entry HTML file
│   ├── postcss.config.js             # PostCSS configuration for Tailwind
│   ├── public/                       # Public assets directory
│   │   └── vite.svg                  # Vite logo
│   ├── src/                      # Source code directory
│   │   ├── App.css                   # Main application styles
│   │   ├── App.jsx                   # Root application component
│   │   ├── assets/                   # Static assets
│   │   ├── components/             # Reusable UI components
│   │   │   ├── Button/               # Button component
│   │   │   ├── ImageView/            # Image viewing component
│   │   │   ├── ProductCard/          # Product display component
│   │   │   └── VideoPlayer/          # Video playback component
│   │   ├── context/                # React context providers
│   │   │   └── AuthContext.jsx       # Authentication context
│   │   ├── hooks/                  # Custom React hooks
│   │   │   ├── hooks.jsx             # General purpose hooks
│   │   │   └── useFetch.jsx          # Data fetching hook
│   │   ├── pages/                  # Application pages
│   │   │   ├── HomePage/             # Home page component
│   │   │   └── Login/                # Login page component
│   │   └── utils/                  # Utility functions
│   │       └── baseApi.jsx           # API configuration
│   ├── tailwind.config.js            # Tailwind CSS configuration
│   └── vite.config.js                # Vite bundler configuration
│
├── server/                   # FastAPI backend
│   ├── controllers/               # Core logic handlers (ML, Video Stream and Other utilities)
│   │   ├── auth.py                   # Authentication controller
│   │   ├── entity_extraction.py      # Entity extraction logic
│   │   ├── freshness_classfier.py    # Freshness classification
│   │   ├── user_controller.py        # User management
│   │   ├── utils.py                  # Utility functions
│   │   └── video_stream.py           # Video streaming handler (Object Detection and Tracking)
│   ├── database/                 # Database related files
│   │   ├── create_db.py              # Database initialization
│   │   ├── database.py               # Database configuration
│   │   ├── models/                   # Database models
│   │   │   └── userModel.py          # User model definition
│   │   └── schemas.py                # Pydantic schemas
│   ├── main.py                       # FastAPI application entry
│   └── routes/                   # API routes
│       ├── streamRoutes.py           # Video Streaming and Processing Endpoints 
│       └── userRoutes.py             # User-related endpoints
│
├── Data/                    # Dataset storage
│   ├── clean_sdf_2400.csv            # Cleaned dataset
│   ├── front.webp                    # Front image asset
│   ├── home_sdf_marketing_sample.csv # Marketing data
│   └── Video/                     # Video dataset directory(contains training andf testing Videos)
│
├── Notebooks/               # Jupyter notebooks (ML EDA , Training and Testing Notebooks)
│   ├── 01_EDA_OCR.ipynb                                                      # Exploratory Data Analysis and GOT OCR2.0 testing 
│   ├── 02b_Yolov11_Object_Detection_Inference.ipynb                          # Inference notebook (for Yolov11 trained model on video )
│   ├── 02_YoloV11_ObjectDetection.ipynb                                      # Training notebook - Yolov11 (MAP50 -98.7 %)
│   ├── 03-freshness-classifier-model.ipynb                                   # Training Notebook - MobileNetv2 (Accuracy -98 %)
│   ├── 04_Architecture_Diagram.ipynb # System architecture
│   ├── Models/             # Trained model files
│   │   ├── mobilenetv2_freshness_classifier.pth                              # Freshness classifier model
│   │   ├── yolov11m_30ep.pt 
│   │   ├── yolov11m_50epv1.pt 
│   │   ├── yolov11m_50epv2.pt
│   │   ├── yolov11m_50epv3.pt                                                # Final Model for Object Detection 
│   │   └── yolov11n_v1.pt   
│   └── runs/                                                                 # Training runs output
│
├── PyScripts/             # Utility Python scripts to Test Models and other utility functions in runtime 
│   ├── analyze_freshness.py                  # Freshness analysis script
│   ├── analyz_multiple_images.py             # Batch image analysis
│   ├── avi2mp4.py                            # Video format converter
│   ├── batch_process_detection_tracking.py   # Batch processing for detection and tracking 
│   ├── detection_ocr.py                      # OCR implementation
│   ├── download_youtube_video.py             # Video downloader
│   ├── freshenss_script.py                   # Freshness checking
│   ├── freshness_classifier.py               # Classifier implementation
│   ├── save_img_objs.py                      # Image object saver along with detection and tracking 
│   ├── save_imgs_without_blur.py             # Image processing
│   ├── test_ocr.py                           # OCR testing
│   ├── verify_cuda.py                        # CUDA verification
│   └── yolo_obj_trackiing.py                 # YOLO tracking test 
│
├── Results/              # Output directory
│   └── Cropped_Images/                       # Processed images
│
├── Submission/                            # Project submission
│   ├── TeamSentinels_Submission.pptx          # Presentation
│   └── TeamSentinelSubmission.pdf             # Documentation
│
└── requirements.txt     # Python dependencies
```