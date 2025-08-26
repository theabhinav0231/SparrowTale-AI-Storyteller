import streamlit as st
import os
import time
import glob

from llm import generate_story
from chunker import process_story_for_multimedia, save_multimedia_data
from image_generation import generate_all_images_from_file
from tts import generate_all_audio_from_file
from movie import Video 

# --- Page Configuration ---
st.set_page_config(
    page_title="Smart Cultural Storyteller",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Helper Functions ---
def cleanup_files():
    """Removes old generated files to keep the space clean for a new run."""
    print("--- Cleaning up old files... ---")
    # Directories for generated assets
    for directory in ["generated_images", "generated_audio", "temp_uploads"]:
        if os.path.exists(directory):
            for f in glob.glob(os.path.join(directory, '*.*')):
                try:
                    os.remove(f)
                except OSError as e:
                    st.error(f"Error removing file {f}: {e}")

    # JSON files, temp files, and final video from the root directory
    for f in glob.glob("multimedia_*.json") + glob.glob("temp_*.*") + glob.glob("*.mp4"):
        if os.path.exists(f):
            os.remove(f)

# --- UI Layout ---
st.title("SparrowTale - 🎬 Smart Cultural Storyteller")
st.markdown("Craft beautiful, culturally rich stories with the power of AI. Provide a prompt, upload a document or audio, and watch your story come to life!")

# Use session state to store the video path and prevent re-runs
if 'video_path' not in st.session_state:
    st.session_state.video_path = None

# Center the main content for a cleaner look
col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    with st.container(border=True):
        # Input widgets
        user_prompt = st.text_area("📝 **Enter your story idea...**", height=150, placeholder="A wise old turtle who teaches a village about patience...")
        
        col_style, col_gender = st.columns(2)
        with col_style:
            story_style = st.selectbox(
                "🎨 **Choose a Story Style**",
                ("Mythical & Folklore", "Historical & Realistic", "Futuristic & Sci-Fi", "Ancient Indian Knowledge")
            )
        with col_gender:
            gender = st.selectbox(
                "🗣️ **Choose a Narrator Voice**",
                ("Female", "Male")
            )

        # File and Audio Uploaders
        doc_file = st.file_uploader("📄 **Upload a document for context (optional)**", type=['txt', 'pdf'])
        audio_file = st.file_uploader("🎤 **Upload an audio prompt (optional)**", type=['wav', 'mp3'])

        # Generate Button
        generate_button = st.button("✨ **Generate Story Video**", use_container_width=True, type="primary")

    # --- Generation Logic ---
    if generate_button:
        # Validate inputs
        if not user_prompt and not doc_file and not audio_file:
            st.error("Please provide a story idea, a document, or an audio prompt to begin.")
        else:
            # 1. Cleanup previous run's files
            cleanup_files()
            st.session_state.video_path = None # Reset video path
            
            # 2. Handle file uploads
            doc_path, audio_path = None, None
            temp_dir = "temp_uploads"
            os.makedirs(temp_dir, exist_ok=True)
            if doc_file:
                doc_path = os.path.join(temp_dir, doc_file.name)
                with open(doc_path, "wb") as f:
                    f.write(doc_file.getbuffer())
            if audio_file:
                audio_path = os.path.join(temp_dir, audio_file.name)
                with open(audio_path, "wb") as f:
                    f.write(audio_file.getbuffer())
                st.info("Audio file uploaded. The text prompt will be ignored.")
                user_prompt = "Transcribe and use the story from the audio file."

            # 3. Main generation pipeline
            with st.spinner("This might take a few minutes... The AI is dreaming up your story... 🌙"):
                progress_bar = st.progress(0, text="Initializing...")
                try:
                    # A. Generate the story text
                    progress_bar.progress(5, text="Step 1/5: Generating the story script...")
                    story_text = generate_story(user_prompt, story_style, audio_path, doc_path)

                    # B. Chunk the story
                    progress_bar.progress(20, text="Step 2/5: Designing the storyboard...")
                    multimedia_data = process_story_for_multimedia(story_text)
                    save_multimedia_data(multimedia_data, "multimedia_data.json")

                    # C. Generate images
                    progress_bar.progress(40, text="Step 3/5: Painting the scenes...")
                    generate_all_images_from_file("multimedia_data.json", output_json_path="multimedia_data_with_images.json")

                    # D. Generate audio narration
                    progress_bar.progress(60, text="Step 4/5: Recording the narration...")
                    generate_all_audio_from_file(
                        "multimedia_data_with_images.json",
                        target_language="English", # This could be made a dropdown too
                        gender=gender.lower(),
                        output_json_path="multimedia_data_final.json"
                    )

                    # E. Stitch everything into a video using your optimized class
                    progress_bar.progress(80, text="Step 5/5: Directing the final movie...")
                    video_creator = Video()
                    final_video_path = "story_video.mp4"
                    success = video_creator.create_video_from_json(
                        "multimedia_data_final.json", 
                        output_filename=final_video_path
                    )
                    
                    if success:
                        st.session_state.video_path = final_video_path
                        progress_bar.progress(100, text="Completed!")
                        st.success("Your story video has been created!")
                    else:
                        st.error("Video creation failed. Please check the logs.")

                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")

    # Display the video if it has been generated
    if st.session_state.video_path and os.path.exists(st.session_state.video_path):
        st.video(st.session_state.video_path)
        
        with open(st.session_state.video_path, "rb") as file:
            st.download_button(
                label="📥 **Download Video**",
                data=file,
                file_name="my_story_video.mp4",
                mime="video/mp4",
                use_container_width=True
            )

# --- Footer ---
st.markdown("---")
st.markdown("Built with ❤️ by Abhinav")