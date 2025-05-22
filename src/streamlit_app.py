"""
Streamlit application for the Accent Detection System.
Integrates video processing and accent detection with a Streamlit web UI.
"""

import os
import sys
import logging
from pathlib import Path
import streamlit as st

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import custom modules
from src.utils.video_processor import VideoProcessor
from src.models.accent_detector import AccentDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create output directory for audio files
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize processors
video_processor = VideoProcessor(output_dir=UPLOAD_FOLDER)
accent_detector = AccentDetector()

def main():
    """Main Streamlit application."""
    
    # Set page title and description
    st.set_page_config(
        page_title="Accent Detection System",
        page_icon="🎙️",
        layout="centered"
    )
    
    st.title("🎙️ Accent Detection System")
    st.markdown("""
    This application detects the accent in spoken English from video URLs.
    Enter a YouTube URL or any other video URL to analyze the speaker's accent.
    """)
    
    # Input form for video URL
    with st.form("video_url_form"):
        video_url = st.text_input("Enter Video URL:", placeholder="https://www.youtube.com/watch?v=...")
        submit_button = st.form_submit_button("Analyze Accent")
    
    # Process the video URL when the form is submitted
    if submit_button and video_url:
        try:
            # Show processing status
            with st.spinner("Processing video..."):
                # Extract audio from video
                st.info("Extracting audio from video...")
                audio_path = video_processor.extract_audio_from_url(video_url)
                
                # Detect accent
                st.info("Detecting accent...")
                result = accent_detector.detect_accent(audio_path)
            
            # Display results
            st.success("Analysis complete!")
            
            # Create columns for better layout
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Detected Accent", result['accent'])
            
            with col2:
                st.metric("Confidence Score", f"{result['confidence_score']:.2f}%")
            
            # Display explanation
            st.subheader("Analysis Explanation")
            st.write(result['explanation'])
            
            # Add a divider
            st.divider()
            
            # Add information about the detected accent
            st.subheader(f"About {result['accent']} English")
            
            # Provide information based on the detected accent
            accent_info = {
                "American": """
                American English is characterized by its rhotic pronunciation (pronouncing 'r' sounds), 
                flat intonation patterns, and distinctive vowel sounds. It's the most widely spoken 
                variety of English globally, heavily influenced by media and entertainment.
                """,
                "British": """
                British English (particularly Received Pronunciation) features non-rhotic pronunciation 
                (dropping 'r' sounds except before vowels), distinctive intonation patterns, and 
                t-glottalization. It's often perceived as formal or prestigious in many parts of the world.
                """,
                "Australian": """
                Australian English is known for its distinctive vowel pronunciation, rising intonation 
                at the end of sentences, and unique vocabulary. It developed from British English but 
                has evolved its own characteristics over time.
                """,
                "Indian": """
                Indian English is influenced by the phonology of Indian languages, with rhythmic patterns 
                that differ from other varieties. It often features stronger consonant emphasis and 
                syllable-timed rhythm rather than stress-timed rhythm.
                """,
                "Canadian": """
                Canadian English shares many features with American English but maintains some British 
                influences. It's known for "Canadian raising" of certain diphthongs and vocabulary 
                that blends American and British terms.
                """,
                "Irish": """
                Irish English (or Hiberno-English) features melodic intonation patterns, distinctive 
                vowel sounds, and grammatical structures influenced by the Irish language. It varies 
                significantly across different regions of Ireland.
                """,
                "Scottish": """
                Scottish English is characterized by its distinctive 'r' pronunciation, unique vowel 
                sounds, and vocabulary influenced by Scots and Scottish Gaelic. It varies considerably 
                across different regions of Scotland.
                """,
                "South African": """
                South African English has been influenced by Afrikaans and indigenous African languages. 
                It features distinctive vowel sounds, intonation patterns, and vocabulary unique to 
                South Africa.
                """,
                "New Zealand": """
                New Zealand English is characterized by its distinctive vowel shift, making it sound 
                different from Australian English despite their proximity. It features unique vocabulary 
                and pronunciation patterns influenced by Māori.
                """,
                "Non-English": """
                The speech appears to be in a language other than English, or the English content is 
                insufficient for accurate accent detection.
                """
            }
            
            st.write(accent_info.get(result['accent'], 
                    "Information about this accent variant is not available in our database."))
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            logger.error(f"Error processing request: {str(e)}")
    
    # Add information about the application
    st.sidebar.title("About")
    st.sidebar.info("""
    This application uses machine learning to detect accents in spoken English.
    
    It works by:
    1. Extracting audio from the provided video URL
    2. Analyzing speech patterns and acoustic features
    3. Classifying the accent based on these features
    
    Supported accents include American, British, Australian, Indian, Canadian, 
    Irish, Scottish, South African, and New Zealand English.
    """)
    
    # Add footer
    st.sidebar.divider()
    st.sidebar.caption("© 2025 Accent Detection System")

if __name__ == "__main__":
    main()
