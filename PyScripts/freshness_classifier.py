import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Set the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: Reinitialize the MobileNetV2 model
# Set pretrained=False since we're loading our own weights
model = models.mobilenet_v2(pretrained=False)

# Step 2: Modify the classifier for binary classification
# The classifier is a Sequential model; we replace the last Linear layer
num_classes = 2  # Number of classes in your dataset (e.g., 'fresh' and 'rotten')
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

# Step 3: Load the saved weights
# Provide the path to your saved model weights
checkpoint_path = "../Notebooks/Models/mobilenetv2_freshness_classifier.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))

# Step 4: Move the model to the device
model = model.to(device)

# Step 5: Set the model to evaluation mode
model.eval()

print("Model reinitialized and weights loaded successfully.")

# Assuming the model has been initialized and loaded as per your provided code
# If not, ensure to run the initialization code before using the classify_image function

def classify_image(image_input, model, device, threshold=0.5):
    """
    Classify an image using the provided model and threshold.

    Parameters:
    - image_input (str or PIL.Image.Image): Path to the image or a PIL Image object.
    - model (torch.nn.Module): The pre-trained and loaded PyTorch model.
    - device (torch.device): The device to run the model on (CPU or GPU).
    - threshold (float): Threshold for classifying as class 1. Default is 0.5.

    Returns:
    - predicted_class (str): The predicted class label ('fresh' or 'rotten').
    - probability (float): The probability of the image belonging to the positive class.
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

    # Load the image
    if isinstance(image_input, str):
        image = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, Image.Image):
        image = image_input.convert('RGB')
    else:
        raise ValueError("image_input must be a file path or a PIL Image.")

    # Apply the preprocessing transformations
    input_tensor = preprocess(image)

    # Create a mini-batch as expected by the model
    input_batch = input_tensor.unsqueeze(0).to(device)  # Shape: [1, 3, 224, 224]

    with torch.no_grad():
        # Get model outputs
        outputs = model(input_batch)  # Shape: [1, num_classes]

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

# Example Usage
if __name__ == "__main__":
    # Path to the image you want to classify
    image_path = "../Results/Cropped_Images/Perishable Grocery_23.png"

    # Define the threshold (adjust as needed)
    threshold = 0.95

    # Classify the image
    predicted_label, probability = classify_image(image_path, model, device, threshold)

    print(f"Predicted Class: {predicted_label}")
    print(f"Probability of 'rotten': {probability:.4f}")
