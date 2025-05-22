"""
Script to test the fixed accent detector with various audio samples.
"""

import os
import sys
import logging
import numpy as np
import soundfile as sf
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the accent detector directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the fixed accent detector
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from models import accent_detector

def generate_test_audio(output_dir, duration=5, sample_rate=16000):
    """
    Generate synthetic audio samples with different characteristics for testing.
    
    Args:
        output_dir: Directory to save the audio files
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
    
    Returns:
        List of generated audio file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define different audio characteristics to test
    test_cases = [
        # American-like characteristics
        {
            "name": "american_test",
            "speech_rate": 0.08,
            "pitch_mean": 90,
            "mfcc_values": [0, 0.6, 0, 0, 0]
        },
        # British-like characteristics
        {
            "name": "british_test",
            "speech_rate": 0.05,
            "pitch_mean": 130,
            "mfcc_values": [0, 0, 0.6, 0, 0]
        },
        # Australian-like characteristics
        {
            "name": "australian_test",
            "speech_rate": 0.06,
            "pitch_mean": 140,
            "mfcc_values": [0, 0, -0.6, 0, 0]
        },
        # Indian-like characteristics
        {
            "name": "indian_test",
            "speech_rate": 0.04,
            "pitch_mean": 85,
            "mfcc_values": [0.6, 0, 0, 0, 0]
        },
        # Canadian-like characteristics
        {
            "name": "canadian_test",
            "speech_rate": 0.075,
            "pitch_mean": 95,
            "mfcc_values": [0, -0.6, 0, 0, 0]
        },
        # Irish-like characteristics
        {
            "name": "irish_test",
            "speech_rate": 0.045,
            "pitch_mean": 110,
            "mfcc_values": [-0.6, 0, 0, 0, 0]
        }
    ]
    
    generated_files = []
    
    for case in test_cases:
        output_path = os.path.join(output_dir, f"{case['name']}.wav")
        
        # Generate white noise as base
        audio = np.random.randn(duration * sample_rate)
        
        # Scale to reasonable amplitude
        audio = audio * 0.1
        
        # Add some tonal components to simulate speech
        if case.get('mfcc_values'):
            for i, val in enumerate(case['mfcc_values'][:5]):
                # Add sine waves with frequencies that would influence MFCCs
                freq = 200 + i * 100  # Different frequencies
                t = np.arange(0, duration, 1/sample_rate)
                audio += val * 0.1 * np.sin(2 * np.pi * freq * t)
        
        # Add pitch variations
        if case.get('pitch_mean', 0) > 0:
            # Simulate pitch by adding a fundamental frequency
            t = np.arange(0, duration, 1/sample_rate)
            audio += 0.2 * np.sin(2 * np.pi * case['pitch_mean'] * t)
        
        # Simulate speech rate by adding amplitude modulation
        if case.get('speech_rate', 0) > 0:
            # Higher speech rate = faster amplitude modulation
            mod_freq = case['speech_rate'] * 10
            t = np.arange(0, duration, 1/sample_rate)
            modulation = 0.5 + 0.5 * np.sin(2 * np.pi * mod_freq * t)
            audio = audio * modulation
        
        # Save the audio file using soundfile instead of deprecated librosa.output
        sf.write(output_path, audio, sample_rate)
        logger.info(f"Generated {output_path}")
        generated_files.append(output_path)
    
    return generated_files

def test_accent_detector(audio_files):
    """
    Test the fixed accent detector with the provided audio files.
    
    Args:
        audio_files: List of audio file paths to test
    
    Returns:
        Dictionary mapping file names to detection results
    """
    # Initialize the fixed accent detector
    detector = accent_detector.AccentDetector(diagnostic_mode=True)
    
    results = {}
    
    # Test each audio file
    for audio_file in audio_files:
        file_name = os.path.basename(audio_file)
        logger.info(f"\nTesting {file_name}...")
        
        try:
            # Detect accent
            result = detector.detect_accent(audio_file)
            
            # Store result
            results[file_name] = result
            
            # Print result
            print(f"File: {file_name}")
            print(f"Detected accent: {result['accent']}")
            print(f"Confidence: {result['confidence_score']:.2f}%")
            print(f"Explanation: {result['explanation']}")
            print("-" * 50)
            
        except Exception as e:
            logger.error(f"Error testing {file_name}: {str(e)}")
            results[file_name] = {
                "accent": "Error",
                "confidence_score": 0.0,
                "explanation": f"Error: {str(e)}"
            }
    
    return results

def analyze_results(results):
    """
    Analyze the test results to verify varied outputs.
    
    Args:
        results: Dictionary mapping file names to detection results
    """
    print("\n--- Analysis of Results ---")
    
    # Check if all results are the same
    accents = [r['accent'] for r in results.values()]
    unique_accents = set(accents)
    
    print(f"Number of test files: {len(results)}")
    print(f"Number of unique accent classifications: {len(unique_accents)}")
    print(f"Unique accents detected: {', '.join(unique_accents)}")
    
    if len(unique_accents) == 1:
        print("ISSUE PERSISTS: All test files were classified with the same accent.")
    elif len(unique_accents) < len(results):
        print("PARTIAL SUCCESS: Some variety in classifications, but not all files have unique accents.")
    else:
        print("SUCCESS: Each test file received a different accent classification.")
    
    # Check confidence scores
    confidence_scores = [r['confidence_score'] for r in results.values()]
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
    
    print(f"Average confidence score: {avg_confidence:.2f}%")
    print(f"Confidence score range: {min(confidence_scores):.2f}% - {max(confidence_scores):.2f}%")

def main():
    """Main test function."""
    # Create test directory
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_audio")
    
    # Generate test audio files
    print("Generating test audio files...")
    audio_files = generate_test_audio(test_dir)
    
    # Test the accent detector
    print("\nTesting accent detector with generated audio files...")
    results = test_accent_detector(audio_files)
    
    # Analyze results
    analyze_results(results)
    
    return results

if __name__ == "__main__":
    main()
