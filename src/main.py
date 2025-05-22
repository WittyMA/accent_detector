"""
Main application file for the Accent Detection System.
Integrates video processing and accent detection with a Flask web UI.
"""

import os
import sys
import logging
from pathlib import Path
from flask import Flask, render_template, request, jsonify, url_for

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import custom modules
from src.utils.video_processor import VideoProcessor
from src.models.accent_detector import AccentDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Create output directory for audio files
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize processors
video_processor = VideoProcessor(output_dir=UPLOAD_FOLDER)
accent_detector = AccentDetector()

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Process video URL and detect accent.
    
    Expects JSON with:
    - video_url: URL of the video to analyze
    
    Returns JSON with:
    - success: Boolean indicating success/failure
    - accent: Detected accent type
    - confidence_score: Confidence score (0-100%)
    - explanation: Brief explanation of the result
    - error: Error message (if any)
    """
    try:
        # Get video URL from request
        data = request.get_json()
        video_url = data.get('video_url')
        
        if not video_url:
            return jsonify({
                'success': False,
                'error': 'No video URL provided'
            }), 400
        
        logger.info(f"Processing video URL: {video_url}")
        
        # Extract audio from video
        audio_path = video_processor.extract_audio_from_url(video_url)
        
        # Detect accent
        result = accent_detector.detect_accent(audio_path)
        
        # Return results
        return jsonify({
            'success': True,
            'accent': result['accent'],
            'confidence_score': round(result['confidence_score'], 2),
            'explanation': result['explanation']
        })
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok'})

# Run the app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
