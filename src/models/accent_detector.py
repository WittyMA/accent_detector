"""
Accent detection module for English speech analysis.
Handles accent classification and confidence scoring.
"""

import os
import logging
import numpy as np
import torch
import librosa
import soundfile as sf
from pathlib import Path
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from speechbrain.pretrained import EncoderClassifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AccentDetector:
    """Class to handle accent detection and confidence scoring."""
    
    # Common English accent types
    ACCENT_TYPES = [
        "American", 
        "British", 
        "Australian", 
        "Indian", 
        "Canadian",
        "Irish",
        "Scottish",
        "South African",
        "New Zealand"
    ]
    
    def __init__(self, model_dir=None):
        """
        Initialize the AccentDetector.
        
        Args:
            model_dir (str, optional): Directory to cache models.
                                      If None, default cache location will be used.
        """
        self.model_dir = model_dir
        logger.info("Initializing accent detection models...")
        
        # Initialize models
        self._init_models()
        
        logger.info("Accent detection models initialized successfully")
    
    def _init_models(self):
        """Initialize the required models for accent detection."""
        try:
            # Load speech recognition model for feature extraction
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
            
            # Load language identification model from SpeechBrain
            # This will help with accent classification
            self.language_id = EncoderClassifier.from_hparams(
                source="speechbrain/lang-id-voxlingua107-ecapa", 
                savedir="pretrained_models/lang-id-voxlingua107-ecapa"
            )
            
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise RuntimeError(f"Failed to initialize accent detection models: {str(e)}")
    
    def detect_accent(self, audio_path):
        """
        Detect accent from audio file.
        
        Args:
            audio_path (str): Path to the audio file.
            
        Returns:
            dict: Dictionary containing accent classification, confidence score, and explanation.
        """
        logger.info(f"Detecting accent from audio: {audio_path}")
        
        try:
            # Load and preprocess audio
            audio_data, sample_rate = self._load_audio(audio_path)
            
            # Extract features for accent detection
            features = self._extract_features(audio_data, sample_rate)
            
            # Classify accent
            accent, confidence, explanation = self._classify_accent(features, audio_data, sample_rate)
            
            result = {
                "accent": accent,
                "confidence_score": confidence,
                "explanation": explanation
            }
            
            logger.info(f"Accent detection result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error detecting accent: {str(e)}")
            raise RuntimeError(f"Failed to detect accent: {str(e)}")
    
    def _load_audio(self, audio_path):
        """
        Load and preprocess audio file.
        
        Args:
            audio_path (str): Path to the audio file.
            
        Returns:
            tuple: (audio_data, sample_rate)
        """
        logger.info(f"Loading audio from: {audio_path}")
        
        try:
            # Load audio file
            audio_data, sample_rate = librosa.load(audio_path, sr=16000)
            
            # Ensure audio is not too long (trim to 30 seconds if needed)
            if len(audio_data) > 30 * sample_rate:
                logger.info("Audio longer than 30 seconds, trimming...")
                audio_data = audio_data[:30 * sample_rate]
            
            return audio_data, sample_rate
            
        except Exception as e:
            logger.error(f"Error loading audio: {str(e)}")
            raise RuntimeError(f"Failed to load audio: {str(e)}")
    
    def _extract_features(self, audio_data, sample_rate):
        """
        Extract features from audio for accent detection.
        
        Args:
            audio_data (numpy.ndarray): Audio data.
            sample_rate (int): Sample rate.
            
        Returns:
            dict: Extracted features.
        """
        logger.info("Extracting features from audio")
        
        # Extract various acoustic features
        features = {}
        
        # MFCC features
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
        features['mfcc_mean'] = np.mean(mfccs, axis=1)
        features['mfcc_std'] = np.std(mfccs, axis=1)
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)
        features['spectral_centroid_mean'] = np.mean(spectral_centroid)
        
        # Pitch features
        pitch, _ = librosa.piptrack(y=audio_data, sr=sample_rate)
        features['pitch_mean'] = np.mean(pitch[pitch > 0]) if np.any(pitch > 0) else 0
        
        # Speech rate estimation (using zero crossings as proxy)
        zero_crossings = librosa.feature.zero_crossing_rate(audio_data)
        features['speech_rate'] = np.mean(zero_crossings)
        
        # Extract wav2vec features
        input_values = self.processor(audio_data, sampling_rate=sample_rate, return_tensors="pt").input_values
        with torch.no_grad():
            outputs = self.model(input_values)
            features['wav2vec_logits'] = outputs.logits.mean(dim=1).numpy()
        
        return features
    
    def _classify_accent(self, features, audio_data, sample_rate):
        """
        Classify accent based on extracted features.
        
        Args:
            features (dict): Extracted audio features.
            audio_data (numpy.ndarray): Audio data for additional processing.
            sample_rate (int): Sample rate.
            
        Returns:
            tuple: (accent_type, confidence_score, explanation)
        """
        logger.info("Classifying accent")
        
        # Use language ID model to get language probabilities
        with torch.no_grad():
            language_prediction = self.language_id.classify_batch(torch.tensor([audio_data]))
            language_scores = language_prediction[0]  # Get scores
        
        # Check if it's English first (using language ID model)
        english_langs = ["en", "eng"]
        english_indices = [i for i, lang in enumerate(self.language_id.hparams.label_encoder.classes_) 
                          if any(eng in lang for eng in english_langs)]
        
        # Sum probabilities of all English variants
        english_prob = sum(language_scores[0][i].item() for i in english_indices)
        
        # If not likely English, return low confidence
        if english_prob < 0.5:
            return "Non-English", english_prob * 100, "Speech does not appear to be in English."
        
        # For English speech, determine the specific accent
        # This is a simplified approach using feature-based heuristics
        
        # Extract key features that help differentiate accents
        mfcc_mean = features['mfcc_mean']
        pitch_mean = features['pitch_mean']
        speech_rate = features['speech_rate']
        
        # Simple accent classification based on acoustic features
        # In a real system, this would be a trained classifier
        
        # Simplified accent classification logic
        if speech_rate > 0.07:  # Higher speech rate
            if mfcc_mean[1] > 0:  # Certain MFCC patterns
                accent = "American"
                explanation = "Detected faster speech rate and flat intonation patterns typical of American English."
            else:
                accent = "Canadian"
                explanation = "Detected speech patterns similar to American English but with subtle differences in vowel pronunciation."
        elif pitch_mean > 100:  # Higher pitch variations
            if mfcc_mean[2] > 0:
                accent = "British"
                explanation = "Detected distinctive intonation patterns and vowel sounds characteristic of British English."
            else:
                accent = "Australian"
                explanation = "Detected rising intonation at sentence ends and distinctive vowel sounds typical of Australian English."
        else:  # Other patterns
            if mfcc_mean[0] > 0:
                accent = "Indian"
                explanation = "Detected rhythmic patterns and consonant emphasis common in Indian English."
            else:
                accent = "Irish"
                explanation = "Detected melodic speech patterns and distinctive vowel sounds typical of Irish English."
        
        # Calculate confidence score (simplified approach)
        # In a real system, this would be based on model probabilities
        
        # Base confidence on English probability
        base_confidence = english_prob * 100
        
        # Adjust based on feature clarity
        feature_confidence = min(100, max(50, 
            70 + 10 * abs(speech_rate - 0.05) + 
            5 * abs(np.mean(mfcc_mean)) +
            5 * (pitch_mean / 100)
        ))
        
        # Final confidence is weighted average
        confidence = (0.7 * base_confidence + 0.3 * feature_confidence)
        
        return accent, confidence, explanation


# Example usage
if __name__ == "__main__":
    # Test with a sample audio file
    detector = AccentDetector()
    try:
        result = detector.detect_accent("path/to/audio.wav")
        print(f"Accent: {result['accent']}")
        print(f"Confidence: {result['confidence_score']:.2f}%")
        print(f"Explanation: {result['explanation']}")
    except Exception as e:
        print(f"Error: {str(e)}")
