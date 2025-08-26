import os
import base64
import json
import streamlit as st
from sarvamai import SarvamAI
from typing import List, Dict, Optional


client = None
try:
    api_key = st.secrets["SARVAM_API_KEY"]
    if api_key:
        client = SarvamAI(api_subscription_key=api_key)
        print("✅ Sarvam AI client for TTS (Bulbul) initialized successfully.")
    else:
        print("⚠️ Warning: SARVAM_API_KEY not found.")
except Exception as e:
    print(f"❌ Error initializing Sarvam AI client: {e}")

# --- Language Mapping ---
LANGUAGE_CODE_MAP = {
    "hindi": "hi-IN", "bengali": "bn-IN", "tamil": "ta-IN", "telugu": "te-IN",
    "gujarati": "gu-IN", "kannada": "kn-IN", "malayalam": "ml-IN", "marathi": "mr-IN",
    "punjabi": "pa-IN", "odia": "od-IN", "english": "en-IN",
}

def get_language_code(language_name: str) -> Optional[str]:
    return LANGUAGE_CODE_MAP.get(language_name.lower())

def generate_audio_from_text(
    text: str,
    language_name: str,
    gender: str,
    output_file_path: str
) -> bool:
    """
    Generates an audio file from a text string using the Sarvam "Bulbul" TTS API
    with a specified gender and pace for the voice.
    """
    if not client: return False
    lang_code = get_language_code(language_name)
    if not lang_code:
        print(f"❌ Language '{language_name}' is not supported. Skipping.")
        return False

    if gender.lower() == "male":
        speaker_name = "abhilash"
        pace_value = 1.0
    else:
        speaker_name = "anushka"
        pace_value = 0.9

    print(f"--- 🎤 Generating audio for chunk: '{text[:50]}...' in {language_name} (Voice: {speaker_name}, Pace: {pace_value}) ---")

    try:
        response = client.text_to_speech.convert(
            text=text,
            model="bulbul:v2",
            target_language_code=lang_code,
            speaker=speaker_name,
            pace=pace_value, # Use the selected pace
            speech_sample_rate=22050,
            enable_preprocessing=True
        )

        combined_audio_b64 = "".join(response.audios)
        audio_data = base64.b64decode(combined_audio_b64)

        with open(output_file_path, "wb") as f:
            f.write(audio_data)

        print(f"✅ Audio saved to {output_file_path}")
        return True

    except Exception as e:
        print(f"❌ An error occurred during the Sarvam TTS API call: {e}")
        return False

def generate_all_audio_from_file(
    json_path: str,
    target_language: str,
    gender: str,
    output_dir: str = "generated_audio",
    output_json_path: str = "multimedia_data_final.json"
) -> List[Dict[str, str]]:
    """
    Reads data from a JSON, generates audio with a specific gender, and saves a final JSON.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            multimedia_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ Error reading or parsing {json_path}: {e}")
        return []

    os.makedirs(output_dir, exist_ok=True)

    for i, item in enumerate(multimedia_data):
        audio_text = item.get("audio_text")
        if not audio_text:
            item["audio_path"] = None
            continue

        file_path = os.path.join(output_dir, f"audio_{i:03d}.mp3")
        success = generate_audio_from_text(audio_text, target_language, gender, file_path)
        item["audio_path"] = file_path if success else None

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(multimedia_data, f, indent=2, ensure_ascii=False)
    print(f"\n--- ✅ Audio generation finished. Final data saved to {output_json_path}. ---")

    return multimedia_data

# Example Usage
# if __name__ == '__main__':
#     json_input_file = "multimedia_data_with_images.json"
#     if not os.path.exists(json_input_file):
#         print(f"❌ Error: Input file '{json_input_file}' not found.")
#         print("Please run image_generation.py first to generate it.")
#     else:
#         target_language_for_story = "English"
#         target_gender_for_story = "male" # Change to "male" to test the other voice

#         generate_all_audio_from_file(
#             json_path=json_input_file,
#             target_language=target_language_for_story,
#             gender=target_gender_for_story
#         )