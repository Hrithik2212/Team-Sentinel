import torch
import torch.nn as nn
from torchvision import models, transforms
import os 
from PIL import Image 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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