"""
Video processing utilities for accent detection system.
Handles video URL input and audio extraction.
"""

import os
import tempfile
import logging
from pathlib import Path
import yt_dlp
import ffmpeg

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoProcessor:
    """Class to handle video URL input and audio extraction."""
    
    def __init__(self, output_dir=None):
        """
        Initialize the VideoProcessor.
        
        Args:
            output_dir (str, optional): Directory to save extracted audio files.
                                       If None, a temporary directory will be used.
        """
        if output_dir:
            self.output_dir = Path(output_dir)
            os.makedirs(self.output_dir, exist_ok=True)
        else:
            self.temp_dir = tempfile.TemporaryDirectory()
            self.output_dir = Path(self.temp_dir.name)
        
        logger.info(f"Output directory set to: {self.output_dir}")
    
    def __del__(self):
        """Clean up temporary directory if it was created."""
        if hasattr(self, 'temp_dir'):
            self.temp_dir.cleanup()
    
    def extract_audio_from_url(self, video_url):
        """
        Extract audio from a video URL.
        
        Args:
            video_url (str): URL of the video to process.
            
        Returns:
            str: Path to the extracted audio file.
            
        Raises:
            ValueError: If the URL is invalid or unsupported.
            RuntimeError: If audio extraction fails.
        """
        logger.info(f"Processing video URL: {video_url}")
        
        # Validate URL
        if not video_url or not isinstance(video_url, str):
            raise ValueError("Invalid URL provided")
        
        try:
            # First, download the video or get direct video file URL
            video_path = self._download_or_get_video(video_url)
            
            # Then extract audio from the video
            audio_path = self._extract_audio(video_path)
            
            return audio_path
            
        except Exception as e:
            logger.error(f"Error processing video URL: {str(e)}")
            raise RuntimeError(f"Failed to process video URL: {str(e)}")
    
    def _download_or_get_video(self, url):
        """
        Download video from URL or get direct file path.
        
        Args:
            url (str): Video URL.
            
        Returns:
            str: Path to the downloaded or direct video file.
        """
        logger.info(f"Downloading or accessing video from: {url}")
        
        # Check if it's a direct file URL (ends with common video extensions)
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        if any(url.lower().endswith(ext) for ext in video_extensions):
            # For direct video URLs, download the file
            output_path = self.output_dir / f"video{Path(url).suffix}"
            
            ydl_opts = {
                'format': 'bestaudio[ext=m4a]/best[ext=mp4]/best',
                'outtmpl': str(output_path),
                'quiet': False,  # Set to False for more verbose output during debugging
                'noplaylist': True,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
                'ignoreerrors': True,  # Continue on download errors
                'no_warnings': False,  # Show warnings for debugging
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            return str(output_path)
        
        # For platform URLs (YouTube, Vimeo, Loom, etc.)
        output_path = self.output_dir / "video.mp4"
        
        # Try multiple format options with fallbacks
        ydl_opts = {
            'format': 'bestaudio[ext=m4a]/best[ext=mp4]/best',  # Prioritize audio formats, then video
            'outtmpl': str(output_path),
            'quiet': False,  # Set to False for more verbose output during debugging
            'noplaylist': True,
            'ignoreerrors': True,  # Continue on download errors
            'no_warnings': False,  # Show warnings for debugging
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
                
            # Check if the file was downloaded successfully
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                # If the file doesn't exist or is empty, try direct audio extraction
                logger.info("Video download failed, trying direct audio extraction...")
                audio_output_path = self.output_dir / "audio.wav"
                
                audio_ydl_opts = {
                    'format': 'bestaudio/best',
                    'outtmpl': str(self.output_dir / "audio"),
                    'quiet': False,
                    'noplaylist': True,
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'wav',
                        'preferredquality': '192',
                    }],
                    'ignoreerrors': False,
                }
                
                with yt_dlp.YoutubeDL(audio_ydl_opts) as ydl:
                    ydl.download([url])
                
                # Check for the extracted audio file
                audio_files = list(self.output_dir.glob("audio.wav"))
                if audio_files:
                    return str(audio_files[0])
                else:
                    raise RuntimeError("Failed to extract audio directly")
        except Exception as e:
            logger.error(f"Error downloading with primary options: {str(e)}")
            
            # Try with simpler options as a last resort
            logger.info("Trying fallback download options...")
            fallback_ydl_opts = {
                'format': 'worstaudio/worst',  # Try worst quality as a last resort
                'outtmpl': str(output_path),
                'quiet': False,
                'noplaylist': True,
            }
            
            with yt_dlp.YoutubeDL(fallback_ydl_opts) as ydl:
                ydl.download([url])
        
        # Check if any file was downloaded
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError("Failed to download video after multiple attempts")
            
        return str(output_path)
    
    def _extract_audio(self, video_path):
        """
        Extract audio from video file.
        
        Args:
            video_path (str): Path to the video file.
            
        Returns:
            str: Path to the extracted audio file.
        """
        logger.info(f"Extracting audio from video: {video_path}")
        
        # Define output audio path
        audio_path = str(self.output_dir / "audio.wav")
        
        try:
            # Extract audio using ffmpeg
            (
                ffmpeg
                .input(video_path)
                .output(audio_path, acodec='pcm_s16le', ac=1, ar='16000')
                .run(quiet=True, overwrite_output=True)
            )
            
            logger.info(f"Audio extracted successfully to: {audio_path}")
            return audio_path
            
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
            
            # Try alternative approach with simpler options
            try:
                logger.info("Trying alternative audio extraction approach...")
                os.system(f'ffmpeg -i "{video_path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "{audio_path}" -y')
                
                if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                    logger.info(f"Audio extracted successfully using alternative approach to: {audio_path}")
                    return audio_path
                else:
                    raise RuntimeError("Alternative audio extraction failed")
            except Exception as alt_e:
                logger.error(f"Alternative extraction error: {str(alt_e)}")
                raise RuntimeError(f"Failed to extract audio: {str(e)}")


# Example usage
if __name__ == "__main__":
    # Test with a sample video URL
    processor = VideoProcessor()
    try:
        audio_file = processor.extract_audio_from_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        print(f"Audio extracted to: {audio_file}")
    except Exception as e:
        print(f"Error: {str(e)}")
