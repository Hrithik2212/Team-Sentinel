import os
import yt_dlp


def download_youtube_video_yt_dlp(video_url, path):
    """
    Downloads a YouTube video using yt-dlp.

    Parameters:
    - video_url (str): The URL of the YouTube video to download.
    - path (str): The directory where the video will be saved.
    """
    # Ensure the target directory exists
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Define yt-dlp options
    ydl_opts = {
        'outtmpl': os.path.join(path, '%(title)s.%(ext)s'),  # Output filename template
        'format': 'bestvideo+bestaudio/best',                # Download best video and audio
        'merge_output_format': 'mp4',                        # Merge into mp4 format
        'quiet': False,                                      # Show download progress
        'no_warnings': True,                                 # Suppress warnings
        'ignoreerrors': False,                               # Stop on download errors
    }
     
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        print("Download completed successfully.")
    except yt_dlp.utils.DownloadError as e:
        print(f"Download error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    video_url = 'https://www.youtube.com/watch?v=rg8y_EkA6a4'
    download_path = '../Data/Video'

    download_youtube_video_yt_dlp(video_url, download_path)
