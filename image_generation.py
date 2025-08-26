import os
import mimetypes
import json
import base64
import streamlit as st
from google import genai
from google.genai import types
from typing import List, Dict, Optional
from PIL import Image
import io
import pathlib


client = None

try:
    api_key = st.secrets["GEMINI_API_KEY"]
    if api_key:
        client = genai.Client(api_key=api_key)
        print("✅ Google AI client for Gemini initialized successfully.")
    else:
        print("⚠️ Warning: GEMINI_API_KEY not found.")
        exit(1)
except Exception as e:
    print(f"❌ Error initializing Google AI client: {e}")
    exit(1)

# --- Helper Functions ---

def save_binary_file(file_name: str, data: bytes):
    try:
        with open(file_name, "wb") as f:
            f.write(data)
        print(f"✅ Image saved to: {file_name}")
    except Exception as e:
        print(f"❌ Error saving file {file_name}: {e}")

def pil_image_to_part(image: Image.Image) -> types.Part:
    """Convert PIL Image to types.Part for use in content."""
    # Convert PIL Image to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()
    
    return types.Part.from_bytes(
        data=img_bytes,
        mime_type='image/jpeg'
    )

def generate_image_with_gemini(
    prompt: str,
    output_file_base: str,
    context_image: Optional[Image.Image] = None
) -> Optional[str]:
    """
    Generates an image, optionally using a previous image as context.
    """
    
    if not client:
        return None
        
    print(f"--- 🎨 Generating image for prompt: '{prompt[:70]}...' ---")
    
    try:
        model = "gemini-2.0-flash-preview-image-generation"
        
        # Build content parts
        content_parts = []
        
        # explicit system prompt that always demands image generation
        if context_image:
            system_prompt = """You are a master storyboard artist creating a visual story sequence. 

            IMPORTANT: You MUST generate an image for every request. 

            Create a visually consistent image that follows the art style and character design of the provided reference image. Maintain consistency in:
            - Character appearance and clothing
            - Art style and color palette  
            - Lighting and atmosphere
            - Overall visual tone

            Style: Cinematic, epic fantasy digital painting with rich details and dramatic lighting.

            Generate an image that illustrates the following scene:"""
            print(" -> Using previous image as context for consistent styling.")
        else:
            system_prompt = """You are a master storyboard artist creating the opening scene of a visual story.

            IMPORTANT: You MUST generate an image for this request.

            Create a stunning, cinematic image in an epic fantasy digital painting style with:
            - Rich, detailed artwork
            - Dramatic lighting and atmosphere
            - High-quality digital painting aesthetic
            - Vivid colors and intricate details

            This is the first scene of the story. Generate an image that illustrates:"""
        
        # Add system prompt
        content_parts.append(types.Part.from_text(text=system_prompt))
        
        # Add context image if provided
        if context_image:
            content_parts.append(pil_image_to_part(context_image))
        
        # Add the actual prompt with explicit image generation instruction
        image_instruction = f"""CREATE AN IMAGE NOW:

        {prompt}

        Remember: You must generate a visual image, not text. Create the artwork described above."""
        
        content_parts.append(types.Part.from_text(text=image_instruction))
        
        # Create content structure
        contents = [
            types.Content(
                role="user",
                parts=content_parts,
            ),
        ]
        
        # Configure generation
        generate_content_config = types.GenerateContentConfig(
            response_modalities=[
                "IMAGE",
                "TEXT",
            ],
        )
        
        # Generate content using streaming
        saved_file_path = None
        text_responses = []
        
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            if (
                chunk.candidates is None
                or chunk.candidates[0].content is None
                or chunk.candidates[0].content.parts is None
            ):
                continue
                
            # Check for image data
            if (chunk.candidates[0].content.parts[0].inline_data and 
                chunk.candidates[0].content.parts[0].inline_data.data):
                
                inline_data = chunk.candidates[0].content.parts[0].inline_data
                data_buffer = inline_data.data
                file_extension = mimetypes.guess_extension(inline_data.mime_type) or ".jpg"
                full_file_name = f"{output_file_base}{file_extension}"
                
                save_binary_file(full_file_name, data_buffer)
                saved_file_path = full_file_name
                print(f"✅ Successfully generated and saved image: {full_file_name}")
                
            # Collect any text responses
            elif hasattr(chunk, 'text') and chunk.text:
                text_responses.append(chunk.text)
        
        # If we got text but no image, print the text for debugging
        if not saved_file_path and text_responses:
            print(f"⚠️ No image generated. API returned text: {' '.join(text_responses)}")
        
        return saved_file_path
        
    except Exception as e:
        print(f"❌ An error occurred during the Gemini API call: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_all_images_from_file(
    json_path: str,
    output_dir: str = "generated_images",
    output_json_path: str = "multimedia_data_with_images.json"
) -> List[Dict[str, str]]:
    """
    Reads data from a JSON, generates images sequentially, and saves a new JSON with image paths.
    """
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            multimedia_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ Error reading or parsing {json_path}: {e}")
        return []
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    previous_image = None
    successful_generations = 0
    
    for i, item in enumerate(multimedia_data):
        print(f"\n{'='*60}")
        print(f"Processing item {i+1}/{len(multimedia_data)} - {'FIRST IMAGE' if i == 0 else 'CONTINUATION'}")
        print(f"{'='*60}")
        
        image_prompt = item.get("image_prompt")
        
        if not image_prompt or "Error:" in image_prompt:
            print(f"⚠️ Skipping item {i}: No valid image prompt found")
            item["image_path"] = None
            continue
        
        # Show the prompt being used
        print(f"📝 Image prompt: {image_prompt[:100]}{'...' if len(image_prompt) > 100 else ''}")
        
        file_base_path = os.path.join(output_dir, f"image_{i:03d}")
        saved_image_path = generate_image_with_gemini(
            image_prompt, 
            file_base_path, 
            context_image=previous_image
        )
        
        item["image_path"] = saved_image_path
        
        # Load the generated image for the next iteration
        if saved_image_path and os.path.exists(saved_image_path):
            try:
                previous_image = Image.open(saved_image_path)
                successful_generations += 1
                print(f"✅ Loaded image {saved_image_path} as context for next generation.")
            except Exception as e:
                print(f"⚠️ Could not load image {saved_image_path} for context. Error: {e}")
                previous_image = None
        else:
            previous_image = None
            print("❌ No image generated for this item!")
        
        # Add a delay between requests
        import time
        time.sleep(2)
    
    # Save the updated JSON with image paths
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(multimedia_data, f, indent=2, ensure_ascii=False)
        print(f"\n--- ✅ Image generation finished. Updated data saved to {output_json_path}. ---")
        print(f"Successfully generated {successful_generations}/{len(multimedia_data)} images.")
        
        # Print summary of results
        print(f"\n📊 GENERATION SUMMARY:")
        for i, item in enumerate(multimedia_data):
            status = "✅ SUCCESS" if item.get("image_path") else "❌ FAILED"
            print(f"  Item {i+1}: {status}")
            
    except Exception as e:
        print(f"❌ Error saving updated JSON: {e}")
    
    return multimedia_data

# Example usage
# if __name__ == '__main__':
#     print("\n--- RUNNING IMAGE GENERATION EXAMPLE ---")
    
#     json_input_file = "multimedia_data.json"
    
#     if not os.path.exists(json_input_file):
#         print(f"❌ Error: Input file '{json_input_file}' not found.")
#         print("Please run chunk.py first to generate it.")
#     else:
#         final_data = generate_all_images_from_file(json_input_file)
        
#         if final_data:
#             print(f"\n--- ✅ Processing completed for {len(final_data)} items ---")
