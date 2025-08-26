import os
from langdetect import detect, LangDetectException
from llm_setup import llm
from prompts import get_story_prompt
from rag_agent import run_rag_agent
from audio_transcription import transcribe_audio_with_auto_detect
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pycountry

if not llm:
    raise ImportError("LLM could not be loaded.")

def detect_language(text: str) -> str:
    """Detects the language of a given text and returns its ISO 639-1 code."""
    try:
        return detect(text)
    except LangDetectException:
        print("Warning: Could not detect language, defaulting to English (en).")
        return "en"

def get_language_name(code: str) -> str:
    """Converts a language code (e.g., 'en') to its full name (e.g., 'English')."""
    try:
        return pycountry.languages.get(alpha_2=code).name
    except AttributeError:
        return "English"

def detect_target_language(user_prompt: str, input_language_name: str) -> str:
    """Uses the LLM to determine the desired output language from the user's prompt."""
    print("--- Detecting target language from prompt... ---")
    prompt_template = PromptTemplate.from_template(
        """Analyze the user's request below. Your task is to determine the desired output language for a story.
        - If the user explicitly mentions a language, return that language name.
        - If the user does NOT explicitly mention an output language, assume they want the story in their input language.
        - Your response MUST be only the name of the language (e.g., "Hindi", "English").
        Input Language: "{input_language}"
        User's Request: "{prompt}"
        Output Language:"""
    )
    chain = prompt_template | llm | StrOutputParser()
    try:
        detected_language = chain.invoke({"prompt": user_prompt, "input_language": input_language_name})
        print(f"✅ LLM detected target language: {detected_language.strip()}")
        return detected_language.strip()
    except Exception as e:
        print(f"Warning: Could not detect target language: {e}. Defaulting to input language.")
        return input_language_name

def generate_story(user_prompt: str, story_style: str, audio_file_path: str = None, doc_file_path: str = None) -> str:
    """
    Orchestrates the full story generation pipeline, only running RAG if a document is provided.
    """
    print("--- Starting Story Generation Pipeline ---")

    input_lang_code = "en"
    if audio_file_path:
        print("Input source: Audio file")
        transcription_result = transcribe_audio_with_auto_detect(audio_file_path)
        user_prompt = transcription_result.get("text")
        detected_lang_code = transcription_result.get("detected_language")
        if not user_prompt or "Error:" in user_prompt:
            return f"Could not generate story due to a transcription error: {user_prompt}"
        if detected_lang_code:
            input_lang_code = detected_lang_code.split('-')[0]
    else:
        input_lang_code = detect_language(user_prompt)
    
    input_language_name = get_language_name(input_lang_code)
    print(f"Detected Input Language: {input_language_name} ({input_lang_code})")

    target_language = detect_target_language(user_prompt, input_language_name)
    rag_context = ""

    if doc_file_path:
        print("\n[Step 1/3] Document provided. Retrieving context with RAG Agent...")
        rag_context = run_rag_agent(user_prompt, file_path=doc_file_path)
        print("✅ Context retrieval complete.")
    else:
        print("\n[Step 1/3] No document provided. Skipping RAG.")

    # The rest of the pipeline proceeds as normal, with or without context.
    print("\n[Step 2/3] Engineering the final prompt...")
    print(f"📋 Story style received: '{story_style}'")
    final_prompt = get_story_prompt(user_prompt, story_style, target_language, rag_context)
    print("✅ Prompt engineering complete.")

        # Add this debug check
    if final_prompt is None:
        print(f"❌ ERROR: get_story_prompt returned None for style: '{story_style}'")
        return f"Error: Invalid story style '{story_style}'. Please select a valid style."
    
    print(f"✅ Prompt engineering complete. Prompt length: {len(final_prompt)} characters")

    print("\n[Step 3/3] Calling the LLM to generate the story...")
    try:
        response = llm.invoke(final_prompt)
        story = response.content
        print("✅ Story generation complete.")
    except Exception as e:
        print(f"❌ An error occurred while calling the LLM: {e}")
        story = f"Error: Could not generate the story. LLM Error: {str(e)}"

    return story

# if __name__ == '__main__':
#     prompt = "write a story about animals and how everybody lived in peace and harmony"
    
#     print("--- RUNNING CROSS-LANGUAGE TEST CASE ---")
#     generated_story = generate_story(
#         user_prompt=prompt,
#         story_style="Indian Wisdom"
#     )
    
#     print("\n\n--- GENERATED STORY ---")
#     print(generated_story)