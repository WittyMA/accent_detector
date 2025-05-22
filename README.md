# Accent Detection System

A machine learning application that detects accents in spoken English from video URLs.

## Overview

This application analyzes speech from videos to identify the speaker's accent. It supports detection of various English accents including American, British, Australian, Indian, Canadian, Irish, Scottish, South African, and New Zealand English.

## Features

- Video URL input (YouTube, Vimeo, direct video links, etc.)
- Audio extraction from videos
- Accent detection with confidence scoring
- Detailed explanation of detected accent characteristics
- User-friendly Streamlit interface
- Docker containerization for easy deployment

## Project Structure

```
accent_detector/
├── Dockerfile                # Docker configuration for containerization
├── docker-compose.yml        # Docker Compose configuration
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── pretrained_models/        # Pre-trained ML models
│   └── lang-id-voxlingua107-ecapa/
├── src/                      # Source code
│   ├── main.py               # Original Flask application
│   ├── streamlit_app.py      # New Streamlit application
│   ├── models/               # ML model implementations
│   │   └── accent_detector.py
│   ├── utils/                # Utility functions
│   │   └── video_processor.py
│   ├── static/               # Static files
│   │   └── uploads/          # Temporary storage for processed files
│   └── templates/            # HTML templates (for Flask version)
│       └── index.html
└── tests/                    # Test files
```

## Installation

### Option 1: Using Docker (Recommended)

1. Make sure you have Docker and Docker Compose installed on your system.
2. Clone or download this repository.
3. Navigate to the project directory.
4. Build and run the Docker container:

```bash
docker-compose up --build
```

5. Access the application at http://localhost:8501

### Option 2: Manual Installation

1. Clone or download this repository.
2. Navigate to the project directory.
3. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

4. Install the required dependencies:

```bash
pip install -r requirements.txt
```

5. Run the Streamlit application:

```bash
streamlit run src/streamlit_app.py
```

6. Access the application at http://localhost:8501

## Usage

1. Open the application in your web browser.
2. Enter a video URL in the input field (YouTube, Vimeo, or direct video link).
3. Click "Analyze Accent" to process the video.
4. View the detected accent, confidence score, and explanation.

## Docker Deployment Instructions

### Building the Docker Image

```bash
docker build -t accent-detector-streamlit .
```

### Running the Docker Container

```bash
docker run -p 8501:8501 accent-detector-streamlit
```

### Using Docker Compose

```bash
docker-compose up
```

To run in detached mode:

```bash
docker-compose up -d
```

To stop the container:

```bash
docker-compose down
```

## Technical Details

- **Video Processing**: Uses yt-dlp and ffmpeg to download videos and extract audio.
- **Audio Analysis**: Utilizes librosa for audio feature extraction.
- **Accent Detection**: Employs SpeechBrain and Transformers models for language identification and accent classification.
- **Web Interface**: Built with Streamlit for an interactive user experience.
- **Containerization**: Docker for consistent deployment across environments.

## Requirements

- Python 3.11+
- FFmpeg
- Required Python packages (see requirements.txt)
- Docker (for containerized deployment)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
