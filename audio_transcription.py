import os
import logging
from typing import Optional, Dict, Any
from sarvamai import SarvamAI
import streamlit as st

class LanguageAgnosticSarvamSTT:
    def __init__(self, api_subscription_key: str):
        if not api_subscription_key:
            raise ValueError("Sarvam API key is missing.")
        self.client = SarvamAI(api_subscription_key=api_subscription_key)
        
        # Common Indian languages supported by Sarvam API
        self.supported_languages = [
            "en-IN", "hi-IN", "gu-IN", "mr-IN", "bn-IN", 
            "ta-IN", "te-IN", "kn-IN", "ml-IN", "pa-IN"
        ]
    
    def _extract_confidence(self, response) -> float:
        """
        Estimates a confidence score from the API response.
        NOTE: This is an estimation because the Sarvam API might not provide a direct
        overall confidence score. A longer, non-empty transcript is considered more confident.
        """
        try:
            if hasattr(response, 'confidence') and response.confidence:
                return response.confidence
            
            # Use response.text as it contains the transcribed string
            if hasattr(response, 'text') and response.text:
                transcript_length = len(response.text.strip())
                # Normalize length to a score between 0.1 and 0.9
                return min(0.9, 0.1 + (transcript_length / 150.0))
            else:
                return 0.0 # No text means zero confidence
        except:
            return 0.0

    def transcribe_with_auto_detection(
        self, 
        audio_file_path: str, 
        model: str = "saarika:v2.5",
        confidence_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Transcribes audio by trying multiple languages and picking the best result.
        """
        best_result_text = ""
        best_confidence = 0.0
        best_language = None
        
        print(f"--- Starting auto-detection transcription for {os.path.basename(audio_file_path)} ---")
        
        for lang_code in self.supported_languages:
            try:
                with open(audio_file_path, "rb") as audio_file:
                    response = self.client.speech_to_text.transcribe(
                        file=audio_file,
                        model=model,
                        language_code=lang_code
                    )
                
                confidence = self._extract_confidence(response)
                print(f"Attempted {lang_code}: Confidence (estimated) = {confidence:.2f}")
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_result_text = response.text.strip()
                    best_language = lang_code
                    
                if confidence >= confidence_threshold:
                    print(f"High confidence result found with {lang_code}. Stopping search.")
                    break
                    
            except Exception as e:
                logging.warning(f"Failed to transcribe with {lang_code}: {e}")
                continue
        
        return {
            "text": best_result_text,
            "detected_language": best_language,
            "confidence": best_confidence,
        }

transcriber_agent = None
try:
    api_key = st.secrets["SARVAM_API_KEY"]
    if api_key:
        transcriber_agent = LanguageAgnosticSarvamSTT(api_key)
        print("✅ Sarvam AI language-agnostic transcriber initialized.")
    else:
        print("⚠️ Warning: SARVAM_API_KEY not found. Transcription will not work.")
except Exception as e:
    print(f"❌ Error initializing Sarvam transcriber: {e}")

def transcribe_audio_with_auto_detect(file_path: str) -> Dict[str, Any]:
    """
    Public function to access the transcription agent.
    """
    if not transcriber_agent:
        return {"text": "Error: Transcriber agent not initialized.", "detected_language": None}

    if not os.path.exists(file_path):
        return {"text": f"Error: Audio file not found at '{file_path}'", "detected_language": None}
    
    return transcriber_agent.transcribe_with_auto_detection(audio_file_path=file_path)