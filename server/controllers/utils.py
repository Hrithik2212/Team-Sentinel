import base64
import cv2
import unicodedata
import subprocess
import io
from pathlib import Path

def normalize_string(s):
    """
    Normalize Unicode characters, remove leading/trailing whitespace,
    and convert to lowercase.
    """
    return unicodedata.normalize('NFKC', s).strip().lower()


# Utility functions
def is_blurry(image, threshold=100.0):
    """Determine if an image is blurry using the variance of the Laplacian."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold, laplacian_var


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