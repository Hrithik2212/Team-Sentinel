import torch
import torch.nn as nn
from torchvision import classifier_models, transforms
from PIL import Image
import requests
import io
import base64
import os
import json

# **Security Note:** It's a best practice to avoid hardcoding API keys in your scripts.
# Consider using environment variables or a secure vault to manage sensitive information.
os.environ["GROQ_API_KEY"] = "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"

# Set the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: Reinitialize the MobileNetV2 classifier_model
# Set pretrained=False since we're loading our own weights
classifier_model = classifier_models.mobilenet_v2(pretrained=False)

# Step 2: Modify the classifier for binary classification
# The classifier is a Sequential classifier_model; we replace the last Linear layer
num_classes = 2  # Number of classes in your dataset (e.g., 'fresh' and 'rotten')
classifier_model.classifier[1] = nn.Linear(classifier_model.classifier[1].in_features, num_classes)

# Step 3: Load the saved weights
# Provide the path to your saved classifier_model weights
checkpoint_path = "../Notebooks/classifier_models/mobilenetv2_freshness_classifier.pth"
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"classifier_model checkpoint not found at {checkpoint_path}")

classifier_model.load_state_dict(torch.load(checkpoint_path, map_location=device))

# Step 4: Move the classifier_model to the device
classifier_model = classifier_model.to(device)

# Step 5: Set the classifier_model to evaluation mode
classifier_model.eval()

print("classifier_model reinitialized and weights loaded successfully.")

# Define the prompt for the Groq API
prompt = """
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
        image.save(buffered, format='PNG')  # Ensure consistency by saving as PNG
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

    # Prepare the API request
    completion = client.chat.completions.create(
        classifier_model="llama-3.2-90b-vision-preview",
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

def process_image(image_url, threshold=0.7):
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
        "threshold": probability   # classifier_model probability
    }

    return result

# Example usage
if __name__ == "__main__":
    # Path to the image you want to classify
    image_path = 'image.png'  # Replace with your actual image path

    # Load the image using PIL
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(json.dumps({"error": f"Failed to open image: {str(e)}"}, indent=4))
        exit(1)

    # Convert image to base64 URL
    try:
        image_url = image_to_base64_url_from_image(image)
    except Exception as e:
        print(json.dumps({"error": f"Failed to convert image to base64 URL: {str(e)}"}, indent=4))
        exit(1)

    # Set the threshold (adjust as needed)
    threshold = 0.7

    # Process the image
    result = process_image(image_url, threshold=threshold)

    # Print the result
    print(json.dumps(result, indent=4))
