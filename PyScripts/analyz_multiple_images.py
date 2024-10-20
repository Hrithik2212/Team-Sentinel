from groq import Groq
import json
from typing import List
import os 
import base64
from pathlib import Path

os.environ["GROQ_API_KEY"] = "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"


def analyze_multiple_images(image_data_urls: List[str], prompt: str) -> dict:
    try:
        # Initialize the Groq client
        client = Groq()
        
        # Prepare the messages list with the prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
        
        # Append each image data URL to the messages content
        for data_url in image_data_urls:
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": data_url
                }
            })
        
        # Make the API request to Groq
        completion = client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
            messages=messages,
            temperature=0,
            max_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
            response_format={"type": "json_object"}  # Ensure response is in JSON format
        )
        
        # Parse and return the JSON response
        response_content = completion.choices[0].message.content
        return json.loads(response_content)
    
    except Exception as e:
        return {"error": str(e)}
    
multi_images_prompt="""
Analyze the images of a single grocery product and extract the following information: 
- Brand name
- Brand details (e.g., logo/tagline)
- Pack size
- Expiry date
- MRP (Maximum Retail Price)
- Product name
- Count/quantity of items - count of the product present in the image 
- Category of the product (e.g., personal care, household items, health supplements, etc.)

Output format: JSON

Handling edge cases:
- If the image is blur, try to identify atleast the brand name, do not leave the brand name N/A as it leads to unpleasent situations.
- If the text is missing, please include 'N/A' in the corresponding field.
- If no visible text or relevant information is present in the image, please return an empty JSON object or a message indicating 'N/A'.
"""



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
image_paths = ['../Data/test.jpg' ]
result = analyze_multiple_images(image_paths , prompt=multi_images_prompt)
print(result)


