from groq import Groq
import json
from server.controllers import utils
import requests
from PIL import Image 
from server.controllers.freshness_classfier import classify_image , device
import io
import os 
import base64


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
                            "url": utils.image_to_base64_url_from_image(image)
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



def perishable_analyze(image_url, classifier_model,threshold=0.9):
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