"""
Script to test non-English language detection in the accent detector.
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

def generate_non_english_test_audio(output_dir, duration=5, sample_rate=16000):
    """
    Generate synthetic audio samples with characteristics of different non-English languages.
    
    Args:
        output_dir: Directory to save the audio files
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
    
    Returns:
        List of generated audio file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define different language characteristics to test
    test_cases = [
        # French-like characteristics
        {
            "name": "french_test",
            "speech_rate": 0.065,
            "pitch_mean": 120,
            "formant_frequencies": [750, 1200, 2500],  # Nasal vowels
            "rhythm": "syllable-timed"
        },
        # German-like characteristics
        {
            "name": "german_test",
            "speech_rate": 0.055,
            "pitch_mean": 100,
            "formant_frequencies": [650, 1700, 2500],  # Front rounded vowels
            "rhythm": "stress-timed"
        },
        # Spanish-like characteristics
        {
            "name": "spanish_test",
            "speech_rate": 0.075,
            "pitch_mean": 110,
            "formant_frequencies": [700, 1300, 2500],  # Clear vowels
            "rhythm": "syllable-timed"
        },
        # Mandarin-like characteristics
        {
            "name": "mandarin_test",
            "speech_rate": 0.06,
            "pitch_mean": 130,
            "formant_frequencies": [800, 1200, 2600],  # Tonal variations
            "rhythm": "syllable-timed"
        },
        # Japanese-like characteristics
        {
            "name": "japanese_test",
            "speech_rate": 0.07,
            "pitch_mean": 125,
            "formant_frequencies": [650, 1200, 2400],  # Mora-timed rhythm
            "rhythm": "mora-timed"
        },
        # Arabic-like characteristics
        {
            "name": "arabic_test",
            "speech_rate": 0.065,
            "pitch_mean": 115,
            "formant_frequencies": [600, 1500, 2700],  # Pharyngealized consonants
            "rhythm": "stress-timed"
        }
    ]
    
    generated_files = []
    
    for case in test_cases:
        output_path = os.path.join(output_dir, f"{case['name']}.wav")
        
        # Generate white noise as base
        audio = np.random.randn(duration * sample_rate)
        
        # Scale to reasonable amplitude
        audio = audio * 0.1
        
        # Add formant frequencies to simulate speech
        if case.get('formant_frequencies'):
            for i, freq in enumerate(case['formant_frequencies']):
                t = np.arange(0, duration, 1/sample_rate)
                # Add formant with decreasing amplitude for higher formants
                amplitude = 0.2 / (i + 1)
                audio += amplitude * np.sin(2 * np.pi * freq * t)
        
        # Add pitch variations
        if case.get('pitch_mean', 0) > 0:
            # Simulate pitch by adding a fundamental frequency
            t = np.arange(0, duration, 1/sample_rate)
            audio += 0.2 * np.sin(2 * np.pi * case['pitch_mean'] * t)
            
            # For tonal languages, add pitch contours
            if case['name'] in ['mandarin_test']:
                # Add rising and falling tones
                for i in range(5):
                    start_time = i * duration / 5
                    end_time = (i + 1) * duration / 5
                    mask = (t >= start_time) & (t < end_time)
                    
                    # Alternate between rising and falling tones
                    if i % 2 == 0:
                        # Rising tone
                        pitch_contour = np.linspace(case['pitch_mean'] * 0.8, case['pitch_mean'] * 1.2, np.sum(mask))
                    else:
                        # Falling tone
                        pitch_contour = np.linspace(case['pitch_mean'] * 1.2, case['pitch_mean'] * 0.8, np.sum(mask))
                    
                    t_segment = t[mask]
                    audio[mask] += 0.15 * np.sin(2 * np.pi * pitch_contour * (t_segment - start_time))
        
        # Simulate speech rhythm
        if case.get('rhythm'):
            # Different rhythm patterns
            if case['rhythm'] == 'syllable-timed':
                # Regular amplitude modulation (syllable-timed languages like Spanish, French)
                mod_freq = 4  # 4 Hz for syllable rhythm
                t = np.arange(0, duration, 1/sample_rate)
                modulation = 0.5 + 0.5 * np.sin(2 * np.pi * mod_freq * t)
                audio = audio * modulation
            elif case['rhythm'] == 'stress-timed':
                # Irregular amplitude modulation (stress-timed languages like German, English)
                t = np.arange(0, duration, 1/sample_rate)
                stress_points = np.random.choice(range(10), 4, replace=False)  # Random stress points
                modulation = np.ones_like(audio) * 0.5
                for sp in stress_points:
                    center = sp * duration / 10
                    width = duration / 20
                    gaussian = np.exp(-0.5 * ((t - center) / width) ** 2)
                    modulation += gaussian * 0.5
                audio = audio * modulation
            elif case['rhythm'] == 'mora-timed':
                # Very regular timing (mora-timed languages like Japanese)
                mod_freq = 7  # Higher frequency for mora rhythm
                t = np.arange(0, duration, 1/sample_rate)
                modulation = 0.5 + 0.5 * np.sin(2 * np.pi * mod_freq * t)
                audio = audio * modulation
        
        # Simulate speech rate by adding amplitude modulation
        if case.get('speech_rate', 0) > 0:
            # Higher speech rate = faster amplitude modulation
            mod_freq = case['speech_rate'] * 10
            t = np.arange(0, duration, 1/sample_rate)
            modulation = 0.5 + 0.5 * np.sin(2 * np.pi * mod_freq * t)
            audio = audio * modulation
        
        # Save the audio file using soundfile
        sf.write(output_path, audio, sample_rate)
        logger.info(f"Generated {output_path}")
        generated_files.append(output_path)
    
    return generated_files

def test_non_english_detection(audio_files):
    """
    Test the accent detector's ability to identify non-English audio.
    
    Args:
        audio_files: List of non-English audio file paths to test
    
    Returns:
        Dictionary mapping file names to detection results
    """
    # Initialize the accent detector
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
            print(f"Detected: {result['accent']}")
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
    Analyze the test results to verify non-English detection.
    
    Args:
        results: Dictionary mapping file names to detection results
    """
    print("\n--- Analysis of Non-English Detection Results ---")
    
    # Count how many were correctly identified as non-English
    non_english_count = sum(1 for r in results.values() if r['accent'] == "Non-English")
    
    print(f"Number of test files: {len(results)}")
    print(f"Number identified as Non-English: {non_english_count}")
    print(f"Detection rate: {non_english_count/len(results)*100:.1f}%")
    
    if non_english_count == len(results):
        print("SUCCESS: All non-English test files were correctly identified as non-English.")
    elif non_english_count > len(results) / 2:
        print("PARTIAL SUCCESS: Majority of non-English test files were correctly identified.")
    else:
        print("ISSUE: Most non-English test files were not correctly identified.")
    
    # Check confidence scores
    confidence_scores = [r['confidence_score'] for r in results.values()]
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
    
    print(f"Average confidence score: {avg_confidence:.2f}%")
    print(f"Confidence score range: {min(confidence_scores):.2f}% - {max(confidence_scores):.2f}%")
    
    # Check which languages were most difficult to identify
    for file_name, result in results.items():
        language = file_name.split('_')[0].capitalize()
        if result['accent'] != "Non-English":
            print(f"Failed to identify {language} as non-English. Detected as: {result['accent']}")

def main():
    """Main test function."""
    # Create test directory
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "non_english_test_audio")
    
    # Generate test audio files
    print("Generating non-English test audio files...")
    audio_files = generate_non_english_test_audio(test_dir)
    
    # Test the accent detector
    print("\nTesting accent detector with non-English audio files...")
    results = test_non_english_detection(audio_files)
    
    # Analyze results
    analyze_results(results)
    
    return results

if __name__ == "__main__":
    main()
