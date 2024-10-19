import subprocess

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

# Example usage
if __name__ == "__main__":
    success = convert_avi_to_mp4("../Results/object_detection_tracking_ocrv1.avi", "../Results/object_detection_tracking_ocrv1.mp4")
    print("Conversion successful" if success else "Conversion failed")