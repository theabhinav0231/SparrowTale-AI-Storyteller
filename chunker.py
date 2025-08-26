import json
import spacy
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict
import os

from llm_setup import llm

if not llm:
    raise ImportError("Creative LLM could not be loaded. Please check llm_setup.py.")

try:
    nlp = spacy.load("en_core_web_sm")
    print("✅ spaCy NLP model loaded successfully.")
except OSError:
    print("❌ spaCy model not found. Please run 'python -m spacy download en_core_web_sm'")
    nlp = None

def chunk_story(story_text: str) -> List[str]:
    """
    Splits a long story text into meaningful chunks based on sentence boundaries.
    """
    if not nlp:
        raise ImportError("spaCy model is not loaded. Cannot chunk story.")

    print("--- Chunking story into meaningful sentence groups... ---")
    
    # Process the text with spaCy to identify sentences
    doc = nlp(story_text)
    sentences = [sent.text.strip() for sent in doc.sents]
    
    chunks = []
    current_chunk = ""
    # Target character count for each chunk, similar to our previous logic
    target_length = 180 

    for sentence in sentences:
        # If adding the new sentence doesn't make the chunk too long, add it
        if len(current_chunk) + len(sentence) + 1 < target_length:
            current_chunk += sentence + " "
        # Otherwise, the current chunk is complete
        else:
            # If the current chunk is not empty, add it to the list
            if current_chunk:
                chunks.append(current_chunk.strip())
            # Start a new chunk with the current sentence
            current_chunk = sentence + " "
            
    # Add the last remaining chunk if it exists
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    print(f"✅ Story split into {len(chunks)} natural chunks.")
    return chunks


def generate_image_prompt(text_chunk: str) -> str:
    print(f"--- Generating image prompt for chunk: '{text_chunk[:50]}...' ---")
    prompt_template = PromptTemplate.from_template(
        """You are an award-winning art director creating a prompt for an AI image generator.
        Translate the following story segment into a single, rich, descriptive visual prompt in English.

        **Instructions:**
        1.  **Style:** 'Epic fantasy digital painting', 'cinematic sci-fi concept art', or 'historical oil painting'.
        2.  **Atmosphere & Lighting:** Describe the mood and lighting.
        3.  **Composition:** Frame the scene (e.g., 'wide-angle shot', 'close-up portrait').
        4.  **Output:** Combine everything into one single paragraph.

        **Story Segment:** "{chunk}"
        **Image Generation Prompt (English):**"""
    )
    chain = prompt_template | llm | StrOutputParser()
    try:
        image_prompt = chain.invoke({"chunk": text_chunk})
        return image_prompt.strip().replace("\n", " ")
    except Exception as e:
        print(f"An error occurred during image prompt generation: {e}")
        return "Error: Could not generate image prompt."


def process_story_for_multimedia(story_text: str) -> List[Dict[str, str]]:
    # ... (This function remains unchanged)
    story_chunks = chunk_story(story_text)
    multimedia_data = []
    for chunk in story_chunks:
        image_prompt = generate_image_prompt(chunk)
        multimedia_data.append({
            "audio_text": chunk,
            "image_prompt": image_prompt
        })
    return multimedia_data


def save_multimedia_data(data: List[Dict[str, str]], output_path: str = "multimedia_data.json"):
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✅ Multimedia data successfully saved to {output_path}")
    except Exception as e:
        print(f"❌ Error saving data to JSON file: {e}")


# Example usage
# if __name__ == '__main__':

#     print("\n--- RUNNING SEMANTIC CHUNKING EXAMPLE ---")
#     sample_story = """
#     In the heart of the celestial expanse, Elara, the moon goddess, resided in a palace woven from starlight and dreams. Her silver hair flowed like a comet's tail, and her eyes held the wisdom of a thousand silent nights. She was the silent guardian.

#     One fateful evening, a shadow crept across the cosmos. It was Helios, the sun god, in his blazing chariot, determined to banish the night forever. He unleashed a torrent of searing light, aiming to shatter Elara's crystalline home. With serene grace, Elara raised her staff. It was not a weapon of war, but an instrument of tranquility. A wave of calming moonlight washed over the sun god's fury, not extinguishing it, but soothing it into a gentle, breathtaking dawn. Helios, humbled, understood that light and dark were not enemies, but eternal partners in the cosmic dance.
#     """
#     multimedia_output = process_story_for_multimedia(sample_story)
    
#     output_file = "multimedia_data.json"
#     save_multimedia_data(multimedia_output, output_file)