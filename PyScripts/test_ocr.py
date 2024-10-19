from groq import Groq
import os 
import base64
from pathlib import Path
import json

os.environ["GROQ_API_KEY"] = "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"

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
- If the image is blur, try to identify atleast the brand name, do not leave the brand name N/A as it leads to unpleasent situations.
- If the text is missing, please include 'N/A' in the corresponding field.
- If no visible text or relevant information is present in the image, please return an empty JSON object or a message indicating 'N/A'.
"""


def analyze_image(image_path):
    # Initialize the Groq client
    client = Groq()

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
                            "url": image_to_base64_url(image_path)
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

def image_to_base64_url(image_path):
    """
    Convert an image file to a base64 URL string.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Base64 URL string of the image
        
    Raises:
        FileNotFoundError: If the image file doesn't exist
        Exception: If there's an error reading the file or encoding it
    """
    try:
        # Convert string path to Path object
        image_path = Path(image_path)
        
        # Check if file exists
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Get the MIME type based on file extension
        mime_type = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
            '.svg': 'image/svg+xml'
        }.get(image_path.suffix.lower(), 'application/octet-stream')
        
        # Read and encode the image file
        with open(image_path, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            
        # Create the complete base64 URL
        base64_url = f"data:{mime_type};base64,{encoded_string}"
        
        return base64_url
        
    except FileNotFoundError as e:
        raise e
    except Exception as e:
        raise Exception(f"Error converting image to base64: {str(e)}")

# Example usage

# Example usage:
image_path = '../Results/Cropped_Images/Packaged Grocery_9.png'
result = analyze_image(image_path)
print(result)


