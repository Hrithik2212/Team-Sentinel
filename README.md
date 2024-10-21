# Smart-Vision-System-Ecommerce-Warehouse

# Demonstration 

* The goal is to immitate conveyor belt and to detect , track , classify , perform OCR and extract entities on ecommerce products moving on a conveyor belt 

<video width="1080" height="720" controls>
  <source src="Submission/Conveyor_Belt_Simulation_Sentinels.mp4" type="video/mp4">
</video>


# Smart Vision System Architecture

The **Smart Vision System** leverages AWS cloud infrastructure and machine learning models to provide real-time video processing, image classification, object detection, object tracking, OCR, and entity extraction. The system is designed to securely stream video from a conveyor belt, analyze the footage for key insights, and display the results to operators through an intuitive dashboard.

![alt text](image.png)


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
