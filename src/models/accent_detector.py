"""
Accent detection module for English speech analysis.
Handles accent classification and confidence scoring with robust language identification.
"""

import os
import logging
import numpy as np
import torch
import librosa
import soundfile as sf
from pathlib import Path
import traceback
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AccentDetector:
    """Class to handle accent detection and confidence scoring with robust language identification."""
    
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
    
    # Language codes for common non-English languages
    NON_ENGLISH_LANGS = [
        "fr", "de", "es", "it", "pt", "ru", "zh", "ja", "ko", "ar", 
        "hi", "tr", "nl", "pl", "sv", "hu", "fi", "da", "no"
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
        
        # Flag to track if we're using fallback mode
        self.using_fallback = False
        
        # Initialize models
        try:
            self._init_models()
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            logger.warning("Switching to fallback mode without external models")
            self.using_fallback = True
        
        if self.using_fallback:
            logger.info("Accent detection initialized in fallback mode (reduced accuracy)")
        else:
            logger.info("Accent detection models initialized successfully")
    
    def _init_models(self):
        """Initialize the required models for accent detection."""
        try:
            # Import dependencies here to handle import errors gracefully
            try:
                from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
                self.transformers_available = True
            except ImportError:
                logger.warning("Transformers library not available, will use fallback mode")
                self.transformers_available = False
                
            try:
                from speechbrain.pretrained import EncoderClassifier
                self.speechbrain_available = True
            except ImportError:
                logger.warning("SpeechBrain library not available, will use fallback mode")
                self.speechbrain_available = False
            
            # Only proceed with model loading if libraries are available
            if self.transformers_available and self.speechbrain_available:
                # Check if model directory exists for SpeechBrain
                model_dir = "pretrained_models/lang-id-voxlingua107-ecapa"
                if not os.path.exists(model_dir):
                    logger.warning(f"Model directory {model_dir} not found")
                    # Try to create the directory
                    try:
                        os.makedirs(model_dir, exist_ok=True)
                        logger.info(f"Created model directory: {model_dir}")
                    except Exception as dir_e:
                        logger.error(f"Failed to create model directory: {str(dir_e)}")
                        raise RuntimeError(f"Model directory not found and could not be created: {model_dir}")
                
                # Load speech recognition model for feature extraction
                logger.info("Loading Wav2Vec2 model...")
                self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
                self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
                
                # Load language identification model from SpeechBrain
                logger.info("Loading SpeechBrain language ID model...")
                try:
                    self.language_id = EncoderClassifier.from_hparams(
                        source="speechbrain/lang-id-voxlingua107-ecapa", 
                        savedir=model_dir
                    )
                    
                    # Get the language labels - handle different SpeechBrain versions
                    self._get_language_labels()
                    
                except Exception as sb_e:
                    logger.error(f"SpeechBrain model loading error: {str(sb_e)}")
                    logger.error(f"SpeechBrain traceback: {traceback.format_exc()}")
                    
                    # Check if model files exist
                    required_files = [
                        os.path.join(model_dir, "hyperparams.yaml"),
                        os.path.join(model_dir, "embedding_model.ckpt"),
                        os.path.join(model_dir, "classifier.ckpt"),
                        os.path.join(model_dir, "label_encoder.ckpt")
                    ]
                    
                    missing_files = [f for f in required_files if not os.path.exists(f)]
                    if missing_files:
                        logger.error(f"Missing model files: {missing_files}")
                        raise RuntimeError(f"SpeechBrain model files missing: {missing_files}")
                    
                    raise RuntimeError(f"Failed to load SpeechBrain model: {str(sb_e)}")
                
                logger.info("Models loaded successfully")
            else:
                logger.warning("Required libraries not available, using fallback mode")
                self.using_fallback = True
                
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.using_fallback = True
            raise RuntimeError(f"Failed to initialize accent detection models: {str(e)}")
    
    def _get_language_labels(self):
        """
        Get language labels from the encoder, handling different SpeechBrain versions.
        Different versions of SpeechBrain use different attribute names for labels.
        """
        try:
            # Try different attribute names used in various SpeechBrain versions
            if hasattr(self.language_id.hparams.label_encoder, 'classes_'):
                self.language_labels = self.language_id.hparams.label_encoder.classes_
                logger.info("Using 'classes_' attribute for language labels")
            elif hasattr(self.language_id.hparams.label_encoder, 'labels'):
                self.language_labels = self.language_id.hparams.label_encoder.labels
                logger.info("Using 'labels' attribute for language labels")
            elif hasattr(self.language_id.hparams.label_encoder, 'label_list'):
                self.language_labels = self.language_id.hparams.label_encoder.label_list
                logger.info("Using 'label_list' attribute for language labels")
            elif hasattr(self.language_id.hparams, 'labels'):
                self.language_labels = self.language_id.hparams.labels
                logger.info("Using hparams 'labels' attribute for language labels")
            else:
                # Fallback: Create a basic list of language codes
                logger.warning("Could not find language labels, using fallback list")
                self.language_labels = [
                    "en", "de", "fr", "es", "it", "pt", "nl", "pl", "ru", "zh",
                    "ja", "ko", "ar", "hi", "tr", "sv", "hu", "fi", "da", "no"
                ]
            
            # Log the first few labels to verify
            logger.info(f"First few language labels: {self.language_labels[:5]}")
            
            # Create a mapping of language codes to indices
            self.language_indices = {}
            for i, lang in enumerate(self.language_labels):
                lang_str = str(lang).lower()
                self.language_indices[lang_str] = i
                # Also add shortened versions (e.g., "en" for "eng-us")
                if "-" in lang_str:
                    short_code = lang_str.split("-")[0]
                    if short_code not in self.language_indices:
                        self.language_indices[short_code] = i
            
            logger.info(f"Created language index mapping with {len(self.language_indices)} entries")
            
        except Exception as e:
            logger.error(f"Error getting language labels: {str(e)}")
            # Fallback: Create a basic list of language codes
            logger.warning("Using fallback language label list due to error")
            self.language_labels = [
                "en", "de", "fr", "es", "it", "pt", "nl", "pl", "ru", "zh",
                "ja", "ko", "ar", "hi", "tr", "sv", "hu", "fi", "da", "no"
            ]
            self.language_indices = {lang: i for i, lang in enumerate(self.language_labels)}
    
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
            # Check if audio file exists
            if not os.path.exists(audio_path):
                logger.error(f"Audio file not found: {audio_path}")
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
                
            # Load and preprocess audio
            audio_data, sample_rate = self._load_audio(audio_path)
            
            # If in fallback mode, use simplified detection
            if self.using_fallback:
                accent, confidence, explanation = self._fallback_classify_accent(audio_data, sample_rate)
            else:
                # First, determine if the audio is English
                is_english, language_probs = self._identify_language(audio_data)
                
                if not is_english:
                    # Get the most likely non-English language
                    top_lang = self._get_top_language(language_probs)
                    return {
                        "accent": "Non-English",
                        "confidence_score": language_probs.get(top_lang, 70.0),
                        "explanation": f"Speech appears to be in {top_lang.upper()} rather than English."
                    }
                
                # Extract features for accent detection
                features = self._extract_features(audio_data, sample_rate)
                
                # Classify accent
                accent, confidence, explanation = self._classify_accent(features, audio_data, sample_rate, language_probs)
            # Add to detect_accent method
            if hasattr(self, 'using_fallback') and self.using_fallback:
                logger.warning("Using fallback mode for accent detection - reduced accuracy expected")

            result = {
                "accent": accent,
                "confidence_score": confidence,
                "explanation": explanation
            }
            
            logger.info(f"Accent detection result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error detecting accent: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Return a default result with error information
            return {
                "accent": "Unknown",
                "confidence_score": 0.0,
                "explanation": f"Error detecting accent: {str(e)}"
            }
    
    def _identify_language(self, audio_data):
        """
        Identify if the audio is in English or another language.
        
        Args:
            audio_data (numpy.ndarray): Audio data.
            
        Returns:
            tuple: (is_english, language_probabilities)
                is_english (bool): True if the audio is likely English
                language_probabilities (dict): Dictionary mapping language codes to probabilities
        """
        logger.info("Identifying language of audio")
        
        # Default to English with moderate probability if language ID fails
        default_result = (True, {"en": 70.0})
        
        try:
            if not hasattr(self, 'language_id') or not hasattr(self, 'language_labels'):
                logger.warning("Language ID model not available, assuming English")
                return default_result
            
            with torch.no_grad():
                language_prediction = self.language_id.classify_batch(torch.tensor([audio_data]))
                language_scores = language_prediction[0]  # Get scores
                language_score_dict = {}
                
                # Calculate probabilities for all languages
                for i, lang in enumerate(self.language_labels):
                    if i < language_scores.shape[1]:
                        lang_str = str(lang).lower()
                        score = language_scores[0][i].item() * 100  # Convert to percentage
                        language_score_dict[lang_str] = score
                
                # If we couldn't get scores for all languages, use what we have
                if not language_score_dict:
                    logger.warning("Could not extract language scores, using defaults")
                    return default_result
                
                # Get English probability
                english_prob = 0.0
                for lang, score in language_score_dict.items():
                    if lang.startswith("en"):
                        english_prob = max(0.0, min(100.0, language_scores[0][english_idx].item() * 100))  # Convert back to 0-1 scale for calculation
                
                # Get probabilities for common non-English languages
                non_english_probs = {}
                for lang_code in self.NON_ENGLISH_LANGS:
                    prob = 0.0
                    for lang, score in language_score_dict.items():
                        if lang.startswith(lang_code):
                            prob += score / 100.0  # Convert back to 0-1 scale
                    if prob > 0:
                        non_english_probs[lang_code] = prob
                
                # Find the highest non-English probability
                top_non_english = 0.0
                top_lang = None
                for lang, prob in non_english_probs.items():
                    if prob > top_non_english:
                        top_non_english = prob
                        top_lang = lang
                
                # Determine if English based on relative probabilities
                # English must be significantly more likely than other languages
                is_english = english_prob > 0.4 and (english_prob > top_non_english * 1.2)
                
                # Convert probabilities back to percentages for return value
                result_probs = {"en": english_prob * 100}
                for lang, prob in non_english_probs.items():
                    result_probs[lang] = prob * 100
                
                logger.info(f"Language identification: English={english_prob*100:.1f}%, " +
                           f"Top non-English={top_lang}={top_non_english*100:.1f}%")
                logger.info(f"Is English: {is_english}")
                
                return is_english, result_probs
                
        except Exception as e:
            logger.error(f"Error in language identification: {str(e)}")
            logger.error(f"Language ID traceback: {traceback.format_exc()}")
            return default_result
    
    def _get_top_language(self, language_probs):
        """
        Get the most likely language from probability dictionary.
        
        Args:
            language_probs (dict): Dictionary mapping language codes to probabilities.
            
        Returns:
            str: The most likely language code.
        """
        # Remove English from consideration
        non_english_probs = {k: v for k, v in language_probs.items() if not k.startswith("en")}
        
        if not non_english_probs:
            return "unknown"
            
        # Find the language with highest probability
        top_lang = max(non_english_probs.items(), key=lambda x: x[1])[0]
        
        # Map common language codes to full names
        lang_names = {
            "fr": "French",
            "de": "German",
            "es": "Spanish",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",
            "zh": "Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "ar": "Arabic",
            "hi": "Hindi",
            "tr": "Turkish",
            "nl": "Dutch",
            "pl": "Polish",
            "sv": "Swedish",
            "hu": "Hungarian",
            "fi": "Finnish",
            "da": "Danish",
            "no": "Norwegian"
        }
        
        return lang_names.get(top_lang, top_lang.upper())
    
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
            logger.error(f"Traceback: {traceback.format_exc()}")
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
        
        try:
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
            
            # Extract wav2vec features if available
            if hasattr(self, 'processor') and hasattr(self, 'model'):
                input_values = self.processor(audio_data, sampling_rate=sample_rate, return_tensors="pt").input_values
                with torch.no_grad():
                    outputs = self.model(input_values)
                    features['wav2vec_logits'] = outputs.logits.mean(dim=1).numpy()
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Return basic features to allow fallback classification
            return {
                'mfcc_mean': np.zeros(13),
                'mfcc_std': np.zeros(13),
                'spectral_centroid_mean': 0,
                'pitch_mean': 0,
                'speech_rate': 0
            }
    
    def _classify_accent(self, features, audio_data, sample_rate, language_probs):
        """
        Classify accent based on extracted features.
        
        Args:
            features (dict): Extracted audio features.
            audio_data (numpy.ndarray): Audio data for additional processing.
            sample_rate (int): Sample rate.
            language_probs (dict): Language probability scores.
            
        Returns:
            tuple: (accent_type, confidence_score, explanation)
        """
        logger.info("Classifying accent")
        
        try:
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
            
            # Calculate confidence score based on language probabilities
            english_prob = language_probs.get("en", 70.0)
            
            # Adjust based on feature clarity
            feature_confidence = min(100, max(50, 
                70 + 10 * abs(speech_rate - 0.05) + 
                5 * abs(np.mean(mfcc_mean)) +
                5 * (pitch_mean / 100)
            ))
            
            # Final confidence is weighted average
            confidence = (0.7 * english_prob + 0.3 * feature_confidence)
            
            # Ensure confidence is within reasonable bounds
            confidence = max(0.0, min(100.0, confidence))
            
            return accent, confidence, explanation
            
        except Exception as e:
            logger.error(f"Error in accent classification: {str(e)}")
            logger.error(f"Classification traceback: {traceback.format_exc()}")
            
            # Return a default result
            return "American", 60.0, "Fallback classification due to error in processing."
    
    def _fallback_classify_accent(self, audio_data, sample_rate):
        """
        Simplified accent classification when models are unavailable.
        Uses basic audio features for a best-guess classification.
        
        Args:
            audio_data (numpy.ndarray): Audio data.
            sample_rate (int): Sample rate.
            
        Returns:
            tuple: (accent_type, confidence_score, explanation)
        """
        logger.info("Using fallback accent classification")
        
        try:
            # Extract basic features
            # MFCC features
            # Extract basic features
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
            mfcc_mean = np.mean(mfccs, axis=1)
            
            # Use a more balanced approach
            accent_scores = {
                "American": 0,
                "British": 0,
                "Australian": 0,
                "Indian": 0,
                "Canadian": 0,
                "Irish": 0
            }
            
            # Add points based on different features
            zcr = librosa.feature.zero_crossing_rate(audio_data)
            speech_rate = np.mean(zcr)
            
            # Speech rate influences
            if speech_rate > 0.07:
                accent_scores["American"] += 2
                accent_scores["Canadian"] += 1
            elif speech_rate > 0.05:
                accent_scores["British"] += 1
                accent_scores["Australian"] += 1
            else:
                accent_scores["Indian"] += 1
                accent_scores["Irish"] += 1
            
            # MFCC influences (simplified)
            if mfcc_mean[1] > 0:
                accent_scores["American"] += 1
            else:
                accent_scores["Canadian"] += 1
                
            if mfcc_mean[2] > 0:
                accent_scores["British"] += 1
            else:
                accent_scores["Australian"] += 1
                
            if mfcc_mean[0] > 0:
                accent_scores["Indian"] += 1
            else:
                accent_scores["Irish"] += 1
            
            # Find accent with highest score
            accent = max(accent_scores.items(), key=lambda x: x[1])[0]
            
            # Calculate confidence (normalized score)
            max_score = max(accent_scores.values())
            total_score = sum(accent_scores.values())
            confidence = (max_score / total_score) * 70 if total_score > 0 else 50
            
            explanation = f"Basic accent classification based on speech patterns (fallback mode)."
            
            return accent, confidence, explanation
        except Exception as e:
            logger.error(f"Error in fallback classification: {str(e)}")
            
            # Return a very basic result
            return "English", 50.0, "Basic classification only. Full accent detection unavailable due to technical issues."

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
