import cv2
import json
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import librosa
import subprocess
from concurrent.futures import ThreadPoolExecutor
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing


def process_single_scene_worker(scene_data):
    """Worker function for parallel scene processing (must be outside class for multiprocessing)"""
    try:
        scene_item, fps, video_size = scene_data
        
        image_path = scene_item.get("image_path")
        audio_path = scene_item.get("audio_path")
        text = scene_item.get("audio_text", "")
        
        # Validate files exist
        if not image_path or not os.path.exists(image_path):
            return None
        if not audio_path or not os.path.exists(audio_path):
            return None
            
        # Get audio duration (simplified for multiprocessing)
        try:
            y, sr = librosa.load(audio_path)
            audio_duration = len(y) / sr
        except:
            try:
                cmd = ['ffprobe', '-i', audio_path, '-show_entries', 'format=duration', 
                       '-v', 'quiet', '-of', 'csv=p=0']
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                audio_duration = float(result.stdout.strip())
            except:
                return None
                
        if audio_duration <= 0:
            return None
            
        total_frames = int(audio_duration * fps)
        
        return {
            'image_path': image_path,
            'audio_path': audio_path,
            'text': text,
            'audio_duration': audio_duration,
            'total_frames': total_frames
        }
        
    except Exception as e:
        print(f"❌ Error in worker processing: {e}")
        return None

def process_scenes_parallel(scenes_data, max_workers=None):
    """Process scenes in parallel using multiprocessing"""
    if max_workers is None:
        # Use 75% of available CPU cores, max 8
        max_workers = min(8, max(1, int(multiprocessing.cpu_count() * 0.75)))
    
    print(f"🔄 Using {max_workers} parallel workers (detected {multiprocessing.cpu_count()} CPU cores)")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_single_scene_worker, scenes_data))
    
    # Filter out None results
    valid_results = [result for result in results if result is not None]
    print(f"✅ Parallel processing completed: {len(valid_results)}/{len(scenes_data)} scenes valid")
    
    return valid_results


class Video:
    def __init__(self):
        self.font_cache = {}
        self.text_overlay_cache = {}
        self.zoom_frame_cache = {}
        
    def create_video_from_json(
        self,
        json_path: str,
        output_filename: str = "story_video.mp4",
        fps: int = 24,
        video_size: tuple = (1920, 1080)
    ):
        """Optimized video creation with caching and pre-processing"""
        
        print(f"📁 Current working directory: {os.getcwd()}")
        print(f"📁 Output video will be saved to: {os.path.abspath(output_filename)}")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                multimedia_data = json.load(f)
            print(f"✅ Successfully loaded JSON with {len(multimedia_data)} items")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"❌ Error reading or parsing {json_path}: {e}")
            return False

        print("--- 🎬 Starting optimized video compilation... ---")
        start_time = time.time()

        if not self.check_ffmpeg():
            return False

        # Pre-process all scenes (this is where the magic happens)
        processed_scenes = self.preprocess_all_scenes(multimedia_data, fps, video_size)
        
        if not processed_scenes:
            print("❌ No valid scenes to process")
            return False

        # Create video much faster now
        return self.create_video_from_preprocessed(processed_scenes, output_filename, fps, video_size, start_time)
    
    def preprocess_all_scenes(self, multimedia_data, fps, video_size):
        """Pre-process all scenes using parallel processing"""
        print("🔄 Pre-processing scenes with parallel processing...")
        
        # Prepare data for parallel processing
        scenes_data = [(item, fps, video_size) for item in multimedia_data]
        
        # Process scenes in parallel
        parallel_results = process_scenes_parallel(scenes_data)
        
        if not parallel_results:
            print("❌ No valid scenes from parallel processing")
            return []
        
        # Now do the heavy processing (text overlays, zoom frames) with results
        processed_scenes = []
        
        for i, scene_result in enumerate(parallel_results):
            try:
                # Pre-create text overlay (once per scene, not per frame!)
                text_overlay = self.create_text_overlay_once(scene_result['text'], video_size)
                
                # Pre-calculate zoom frames (create multiple zoom levels)
                zoom_frames = self.precalculate_zoom_frames(
                    scene_result['image_path'], 
                    scene_result['total_frames'], 
                    video_size
                )
                
                processed_scenes.append({
                    'audio_path': scene_result['audio_path'],
                    'text_overlay': text_overlay,
                    'zoom_frames': zoom_frames,
                    'total_frames': scene_result['total_frames'],
                    'duration': scene_result['audio_duration']
                })
                
                print(f"✅ Pre-processed scene {i}: {scene_result['audio_duration']:.2f}s ({scene_result['total_frames']} frames)")
                
            except Exception as e:
                print(f"⚠️ Error pre-processing scene {i}: {e}")
                
        return processed_scenes
    
    def create_text_overlay_once(self, text, video_size):
        """Create text overlay once and reuse for all frames"""
        if not text.strip():
            return None
            
        # Check cache first
        cache_key = f"{text}_{video_size[0]}_{video_size[1]}"
        if cache_key in self.text_overlay_cache:
            return self.text_overlay_cache[cache_key]
        
        # Create transparent overlay with text
        overlay = Image.new('RGBA', video_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Load font once
        font_size = max(32, video_size[0] // 40)
        font = self.get_cached_font(font_size)
        
        # Calculate text positioning
        max_width = int(video_size[0] * 0.75)
        wrapped_text = self.wrap_text(text, font, max_width)
        
        text_bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        x = (video_size[0] - text_width) // 2
        y = video_size[1] - text_height - 60
        
        # Draw styled background and text
        self.draw_modern_style(draw, wrapped_text, font, x, y, text_width, text_height)
        
        # Cache and return
        self.text_overlay_cache[cache_key] = overlay
        return overlay
    
    def precalculate_zoom_frames(self, image_path, total_frames, video_size):
        """Pre-calculate all zoom frames for Ken Burns effect"""
        cache_key = f"{image_path}_{total_frames}_{video_size[0]}_{video_size[1]}"
        if cache_key in self.zoom_frame_cache:
            return self.zoom_frame_cache[cache_key]
        
        # Load and prepare base image
        img = cv2.imread(image_path)
        if img is None:
            img = np.zeros((video_size[1], video_size[0], 3), dtype=np.uint8)
        
        # Pre-calculate zoom levels (sample every few frames instead of all)
        zoom_start = 1.0
        zoom_end = 1.15
        sample_rate = 4  # Calculate every 4th frame, interpolate others
        
        zoom_frames = {}
        for frame_num in range(0, total_frames, sample_rate):
            progress = frame_num / total_frames if total_frames > 0 else 0
            current_zoom = zoom_start + (zoom_end - zoom_start) * progress
            zoomed_frame = self.apply_zoom_effect_fast(img, current_zoom, video_size)
            zoom_frames[frame_num] = zoomed_frame
        
        # Cache the result
        self.zoom_frame_cache[cache_key] = zoom_frames
        return zoom_frames
    
    def apply_zoom_effect_fast(self, img, zoom_factor, target_size):
        """Faster zoom effect without per-frame calculations"""
        if zoom_factor == 1.0:
            return cv2.resize(img, target_size)
        
        target_w, target_h = target_size
        new_w = int(target_w * zoom_factor)
        new_h = int(target_h * zoom_factor)
        
        # Resize once
        zoomed = cv2.resize(img, (new_w, new_h))
        
        # Crop to center
        start_x = max(0, (new_w - target_w) // 2)
        start_y = max(0, (new_h - target_h) // 2)
        
        return zoomed[start_y:start_y + target_h, start_x:start_x + target_w]
    
    def create_video_from_preprocessed(self, processed_scenes, output_filename, fps, video_size, start_time):
        """Create video from pre-processed scenes (much faster)"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_video = "temp_video.mp4"
        video_writer = cv2.VideoWriter(temp_video, fourcc, fps, video_size)
        
        if not video_writer.isOpened():
            print("❌ Failed to initialize video writer")
            return False
        
        temp_audio_files = []
        sample_rate = 4  # Use every 4th precalculated frame
        
        for i, scene in enumerate(processed_scenes):
            print(f"🎬 Writing scene {i}: {scene['total_frames']} frames")
            
            for frame_num in range(scene['total_frames']):
                # Get the closest pre-calculated zoom frame
                closest_zoom_frame = self.get_closest_zoom_frame(
                    scene['zoom_frames'], frame_num, sample_rate
                )
                
                # Composite text overlay if exists (fast alpha blending)
                if scene['text_overlay']:
                    final_frame = self.fast_composite(closest_zoom_frame, scene['text_overlay'])
                else:
                    final_frame = closest_zoom_frame
                
                video_writer.write(final_frame)
            
            temp_audio_files.append(scene['audio_path'])
            print(f"✅ Scene {i} completed")
        
        video_writer.release()
        cv2.destroyAllWindows()
        
        # Continue with audio processing...
        return self.finalize_video(temp_video, temp_audio_files, output_filename, start_time)
    
    def get_closest_zoom_frame(self, zoom_frames, frame_num, sample_rate):
        """Get the closest pre-calculated zoom frame"""
        closest_key = (frame_num // sample_rate) * sample_rate
        if closest_key in zoom_frames:
            return zoom_frames[closest_key]
        
        # Find the nearest available frame
        available_keys = sorted(zoom_frames.keys())
        closest_key = min(available_keys, key=lambda x: abs(x - frame_num))
        return zoom_frames[closest_key]
    
    def fast_composite(self, background, overlay):
        """Fast alpha compositing using OpenCV"""
        # Convert PIL overlay to OpenCV format
        overlay_cv = cv2.cvtColor(np.array(overlay), cv2.COLOR_RGBA2BGRA)
        
        # Extract alpha channel
        alpha = overlay_cv[:, :, 3] / 255.0
        
        # Blend channels
        for c in range(3):
            background[:, :, c] = (alpha * overlay_cv[:, :, c] + 
                                 (1 - alpha) * background[:, :, c])
        
        return background
    
    def get_cached_font(self, font_size):
        """Get font from cache or load once"""
        if font_size not in self.font_cache:
            font_candidates = [
                "C:/Windows/Fonts/segoeui.ttf",
                "C:/Windows/Fonts/arial.ttf",
                "arial.ttf"
            ]
            
            for font_path in font_candidates:
                try:
                    self.font_cache[font_size] = ImageFont.truetype(font_path, font_size)
                    break
                except:
                    continue
            else:
                self.font_cache[font_size] = ImageFont.load_default()
        
        return self.font_cache[font_size]
    
    def draw_modern_style(self, draw, text, font, x, y, text_width, text_height):
        """Draw modern style text (optimized)"""
        padding = 25
        corner_radius = 15
        
        # Background with rounded corners
        bg_coords = [x - padding, y - padding, x + text_width + padding, y + text_height + padding]
        draw.rounded_rectangle(bg_coords, radius=corner_radius, fill=(0, 0, 0, 200))
        
        # Text
        draw.multiline_text((x, y), text, font=font, fill=(255, 255, 255, 255), align='center')
    
    def finalize_video(self, temp_video, temp_audio_files, output_filename, start_time):
        """Finalize video with audio"""
        if not temp_audio_files:
            print("❌ No audio files to process")
            return False
        
        print("🔊 Processing audio...")
        final_audio = "temp_combined_audio.wav"
        
        if not self.combine_audio_files(temp_audio_files, final_audio):
            return False
        
        print("🎬 Finalizing video...")
        if not self.merge_video_audio(temp_video, final_audio, output_filename):
            return False
        
        if os.path.exists(output_filename):
            file_size = os.path.getsize(output_filename)
            total_time = time.time() - start_time
            
            print(f"🎉 SUCCESS! Video created in {total_time:.1f} seconds")
            print(f"📊 File size: {file_size:,} bytes")
            print(f"📹 Location: {os.path.abspath(output_filename)}")
            
            # Cleanup
            self.cleanup_temp_files([temp_video, final_audio])
            return True
        
        return False

    # ===== ALL YOUR ORIGINAL HELPER METHODS =====
    
    def check_ffmpeg(self):
        """Check if FFmpeg is available"""
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("✅ FFmpeg is available")
                return True
            else:
                print("❌ FFmpeg not working properly")
                print(f"Error: {result.stderr}")
                return False
        except FileNotFoundError:
            print("❌ FFmpeg not found in PATH")
            return False
        except subprocess.TimeoutExpired:
            print("❌ FFmpeg check timed out")
            return False

    def get_audio_duration(self, audio_path: str) -> float:
        """Get duration of audio file using librosa with fallback to FFprobe"""
        try:
            y, sr = librosa.load(audio_path)
            duration = len(y) / sr
            print(f"🔊 Audio duration (librosa): {duration:.2f}s")
            return duration
        except Exception as e:
            print(f"⚠️ Librosa failed, trying FFprobe: {e}")
            try:
                # Fallback method using FFmpeg
                cmd = ['ffprobe', '-i', audio_path, '-show_entries', 'format=duration', 
                       '-v', 'quiet', '-of', 'csv=p=0']
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    duration = float(result.stdout.strip())
                    print(f"🔊 Audio duration (ffprobe): {duration:.2f}s")
                    return duration
                else:
                    print(f"❌ FFprobe failed: {result.stderr}")
                    return 0
            except Exception as e2:
                print(f"❌ Both librosa and FFprobe failed: {e2}")
                return 0

    def wrap_text(self, text: str, font, max_width: int) -> str:
        """Wrap text to fit within specified width"""
        try:
            words = text.split()
            lines = []
            current_line = []

            for word in words:
                test_line = ' '.join(current_line + [word])
                # Use a temporary draw object to measure text
                temp_img = Image.new('RGB', (1, 1))
                temp_draw = ImageDraw.Draw(temp_img)
                bbox = temp_draw.textbbox((0, 0), test_line, font=font)

                if bbox[2] <= max_width:
                    current_line.append(word)
                else:
                    if current_line:
                        lines.append(' '.join(current_line))
                        current_line = [word]
                    else:
                        lines.append(word)  # Single word is too long, add anyway

            if current_line:
                lines.append(' '.join(current_line))

            return '\n'.join(lines)
        except Exception as e:
            print(f"❌ Error wrapping text: {e}")
            return text

    def combine_audio_files(self, audio_files: list, output_path: str) -> bool:
        """Combine multiple audio files into one WAV file"""
        try:
            if len(audio_files) == 1:
                # Convert single file to WAV format
                cmd = ['ffmpeg', '-i', audio_files[0], '-c:a', 'pcm_s16le', output_path, '-y']
                print(f"🔧 Converting single audio file: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            else:
                # Create a file list for concatenation
                list_file = 'audio_list.txt'
                with open(list_file, 'w', encoding='utf-8') as f:
                    for audio_file in audio_files:
                        # Use forward slashes and escape single quotes
                        safe_path = audio_file.replace("'", "'\"'\"'").replace('\\', '/')
                        f.write(f"file '{safe_path}'\n")

                # Concatenate audio files to WAV
                cmd = [
                    'ffmpeg', '-f', 'concat', '-safe', '0', '-i', list_file,
                    '-c:a', 'pcm_s16le', output_path, '-y'
                ]
                print(f"🔧 Concatenating audio files: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                
                # Clean up list file
                if os.path.exists(list_file):
                    os.remove(list_file)

            if result.returncode != 0:
                print(f"❌ Audio combination failed with return code {result.returncode}")
                print(f"📄 STDERR: {result.stderr}")
                return False
            else:
                print(f"✅ Audio combined successfully: {os.path.getsize(output_path):,} bytes")
                return True

        except Exception as e:
            print(f"❌ Exception during audio combination: {e}")
            return False

    def merge_video_audio(self, video_path: str, audio_path: str, output_path: str) -> bool:
        """Merge video and audio using FFmpeg with error checking"""
        try:
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-i', audio_path,
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-strict', 'experimental',
                '-shortest',  # Match shortest stream
                output_path,
                '-y'  # Overwrite output file
            ]

            print(f"🔧 Running FFmpeg merge: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                print(f"❌ FFmpeg merge failed with return code {result.returncode}")
                print(f"📄 STDOUT: {result.stdout}")
                print(f"📄 STDERR: {result.stderr}")
                return False
            else:
                print(f"✅ FFmpeg merge succeeded!")
                return True

        except subprocess.TimeoutExpired:
            print("❌ FFmpeg merge timed out after 5 minutes")
            return False
        except Exception as e:
            print(f"❌ Exception during video merge: {e}")
            return False

    def cleanup_temp_files(self, file_list: list):
        """Clean up temporary files"""
        for file_path in file_list:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"🗑️ Cleaned up: {file_path}")
                except Exception as e:
                    print(f"⚠️ Could not remove {file_path}: {e}")

# Example usage
# if __name__ == '__main__':
#     optimizer = Video()
#     final_json_file = "multimedia_data_final.json"
    
#     if not os.path.exists(final_json_file):
#         print(f"❌ Final data file '{final_json_file}' not found.")
#         print("Please run chunk.py -> image_generation.py -> tts.py to create it.")
#     else:
#         success = optimizer.create_video_from_json(final_json_file)
#         if success:
#             print("\n🎉 Optimized video creation completed!")
#         else:
#             print("\n❌ Video creation failed. Check the error messages above.")
