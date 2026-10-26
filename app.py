import streamlit as st
import tempfile
import cv2
import numpy as np
import pandas as pd
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from project_utils.video import VideoProcessor
from project_utils.video_io import AsyncVideoWriter, GPUVideoReader, create_async_writer, create_h264_writer
from project_utils.parallel_processor import process_video_concurrent
from engine.tracker import YOLOTracker, TrackNetTracker
from engine.geometry import GeometryEngine
from engine.analysis import AnalysisEngine
from engine.gemini_coach import GeminiCoach
from streamlit_drawable_canvas import st_canvas
from project_utils.security import save_uploaded_file_securely
from project_utils import ui_theme as ui
from project_utils.results_view import render_results

# --- MONKEYPATCH REMOVED (Downgraded Streamlit) ---

# --- Cached Model Loaders (eliminates 3-5s startup on reruns) ---
@st.cache_resource
def load_yolo_tracker():
    """Load YOLO tracker with TensorRT (cached across reruns)."""
    return YOLOTracker()

@st.cache_resource
def load_tracknet_tracker():
    """Load TrackNet + InpaintNet (cached across reruns)."""
    return TrackNetTracker()

st.set_page_config(page_title="Badminton AI", page_icon="🏸", layout="wide")
ui.inject_theme()
ui.header()

# --- session state ---
if 'corners' not in st.session_state:
    st.session_state.corners = None
if 'coach_insights' not in st.session_state:
    st.session_state.coach_insights = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'processing_error' not in st.session_state:
    st.session_state.processing_error = None

# Palette: Toast notification for completed analysis or errors
if st.session_state.get('processing_complete'):
    st.toast("Analysis complete! Results are ready.", icon="✅")
    st.session_state.processing_complete = False

if st.session_state.get('processing_error'):
    st.toast(f"AI Coach Warning: {st.session_state.processing_error}", icon="⚠️")
    st.session_state.processing_error = None

# --- sidebar ---
st.sidebar.header("Configuration")

# Check Device
import torch
device_name = "🟢 GPU (CUDA)" if torch.cuda.is_available() else "🟠 CPU"
st.sidebar.markdown(f"**Processor:** {device_name}")
if not torch.cuda.is_available():
    st.sidebar.caption("Use `supergfxctl` to enable RTX 4080 if available.")


# API Keys
st.sidebar.divider()
st.sidebar.markdown("### 🤖 AI Coach")
_typed_key = st.sidebar.text_input("Gemini API Key", type="password", help="Needed for the AI Coach tab. Get one at https://aistudio.google.com/app/apikey")
_env_key = os.environ.get("GEMINI_API_KEY", "")
gemini_api_key = _typed_key or _env_key
if _env_key and not _typed_key:
    st.sidebar.caption("✅ Using GEMINI_API_KEY from the environment")

uploaded_file = st.sidebar.file_uploader(
    "Upload Video",
    type=["mp4", "mov", "avi", "webm", "mkv"],
    help="Limit 1GB. Supported formats: MP4, MOV, AVI, WebM, MKV"
)

# --- main ---
if uploaded_file:
    ui.stepper(3 if st.session_state.get('analysis_results') else 1)
if uploaded_file:
    # Use name + size as a simple unique identifier for the session
    file_id = f"{uploaded_file.name}-{uploaded_file.size}"

    # Initialize session state variables for file tracking if needed
    if 'current_file_id' not in st.session_state:
        st.session_state.current_file_id = None
    if 'video_path' not in st.session_state:
        st.session_state.video_path = None
    if 'transcoded_path' not in st.session_state:
        st.session_state.transcoded_path = None

    # Check if this is a new file or the same one
    if file_id != st.session_state.current_file_id:
        # Cleanup OLD artifacts if they exist
        if st.session_state.video_path and os.path.exists(st.session_state.video_path):
            try:
                os.remove(st.session_state.video_path)
            except Exception:
                pass
        if st.session_state.transcoded_path and os.path.exists(st.session_state.transcoded_path):
            try:
                os.remove(st.session_state.transcoded_path)
            except Exception:
                pass

        # Also clean up previous analysis results if we switch videos
        if st.session_state.get('analysis_results'):
            old_results = st.session_state['analysis_results']
            if 'clips_dir' in old_results and os.path.exists(old_results['clips_dir']):
                try:
                    shutil.rmtree(old_results['clips_dir'])
                except Exception:
                    pass
            # Cleanup converted file from old analysis
            if 'converted_file' in old_results and os.path.exists(old_results['converted_file']):
                try:
                    os.remove(old_results['converted_file'])
                except Exception:
                    pass
            st.session_state.analysis_results = None
            st.session_state.corners = None
            st.session_state.coach_insights = None

        # Save NEW file
        video_path = save_uploaded_file_securely(
            uploaded_file,
            allowed_extensions=["mp4", "mov", "avi", "webm", "mkv"],
            max_size_mb=1024
        )
        if not video_path:
            st.error("Error saving file. It might exceed the size limit or have an invalid extension.")
            st.stop()

        st.session_state.video_path = video_path
        st.session_state.current_file_id = file_id
        st.session_state.transcoded_path = None # Reset

        # Reset calibration frame
        if 'calibration_frame_idx' in st.session_state:
            st.session_state.calibration_frame_idx = 0

        original_suffix = os.path.splitext(video_path)[1].lower()

        # Pre-transcode WebM/MKV files (often use AV1 codec not supported by OpenCV)
        if original_suffix in ['.webm', '.mkv']:
            st.info("Transcoding WebM/MKV to H.264 for compatibility...")
            t_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            transcode_cmd = [
                "ffmpeg", "-y", "-i", video_path, "-c:v", "libx264", "-preset", "fast",
                "-crf", "23", "-c:a", "aac", t_path
            ]
            result = subprocess.run(transcode_cmd, stderr=subprocess.DEVNULL).returncode
            if result == 0 and os.path.exists(t_path) and os.path.getsize(t_path) > 0:
                st.session_state.transcoded_path = t_path
                st.success("Transcoding complete!")
            else:
                st.error("Transcoding failed. Please try a different video format.")

    # Use cached paths
    video_path = st.session_state.transcoded_path if st.session_state.transcoded_path else st.session_state.video_path
    
    vp = VideoProcessor()
    info = vp.get_video_info(video_path)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("1. Court Calibration")
        
        # Frame Selector for Calibration
        st.markdown("**Select Calibration Frame** (Skip intros)")

        # Initialize session state for fine-tuning
        if 'calibration_frame_idx' not in st.session_state:
            st.session_state.calibration_frame_idx = 0

        # Ensure value is within bounds
        max_frames = info['frame_count'] - 1
        if st.session_state.calibration_frame_idx > max_frames:
            st.session_state.calibration_frame_idx = 0

        start_frame_idx = st.slider(
            "Frame Index",
            0,
            max_frames,
            key="calibration_frame_idx",
            help="Select a frame where the court boundaries are clearly visible for calibration."
        )
        
        # Palette: Fine-tune controls (+/- 1 second & 1 frame)
        c_fast_back, c_back, c_fwd, c_fast_fwd, _ = st.columns([1, 1, 1, 1, 3])
        step_size_sec = int(info['fps']) # Jump 1 second

        with c_fast_back:
            if st.button("⏪ -1s", use_container_width=True, help="Go back 1 second"):
                st.session_state.calibration_frame_idx = max(0, st.session_state.calibration_frame_idx - step_size_sec)
                st.rerun()

        with c_back:
            if st.button("◀ -1f", use_container_width=True, help="Go back 1 frame"):
                st.session_state.calibration_frame_idx = max(0, st.session_state.calibration_frame_idx - 1)
                st.rerun()

        with c_fwd:
            if st.button("▶ +1f", use_container_width=True, help="Go forward 1 frame"):
                st.session_state.calibration_frame_idx = min(max_frames, st.session_state.calibration_frame_idx + 1)
                st.rerun()

        with c_fast_fwd:
            if st.button("⏩ +1s", use_container_width=True, help="Go forward 1 second"):
                st.session_state.calibration_frame_idx = min(max_frames, st.session_state.calibration_frame_idx + step_size_sec)
                st.rerun()

        # Palette: Show timestamp for better UX
        timestamp = start_frame_idx / info['fps']
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)
        st.caption(f"⏱️ Video Time: {minutes:02d}:{seconds:02d}")

        # Get specific frame
        frame_for_calib = vp.get_frame(video_path, start_frame_idx)
        
        if frame_for_calib is not None:
            height, width, _ = frame_for_calib.shape
            
            # Canvas for Calibration
            st.info("💡 **Drag the 4 GREEN points** to the corners of the Singles Court (inner lines). The **Green Box** connects them.")
            
            # Create a canvas
            # We resize image to fit column to avoid scrolling if too big
            canvas_width = 700
            canvas_height = int(height * (canvas_width / width))
            
            # Initialize session state for corners if not present
            if 'calibration_corners' not in st.session_state:
                # Default: 20%, 80% inset
                st.session_state['calibration_corners'] = [
                    (int(canvas_width * 0.2), int(canvas_height * 0.2)), # TL
                    (int(canvas_width * 0.8), int(canvas_height * 0.2)), # TR
                    (int(canvas_width * 0.8), int(canvas_height * 0.8)), # BR
                    (int(canvas_width * 0.2), int(canvas_height * 0.8))  # BL
                ]

            # Palette: Initialize canvas version for reset capability
            if 'canvas_version' not in st.session_state:
                st.session_state.canvas_version = 0

            # Prepare Background Image with Lines connecting the CURRENT corners
            # We draw on a copy of the resized frame
            bg_image = cv2.resize(frame_for_calib, (canvas_width, canvas_height))
            
            # Draw lines based on session_state (which reflects last known position)
            pts = np.array(st.session_state['calibration_corners'], np.int32)
            # Reshape for polylines (points must be (N, 1, 2))
            pts = pts.reshape((-1, 1, 2))
            # Draw green box
            cv2.polylines(bg_image, [pts], True, (0, 255, 0), 1)

            # Initial drawing with 4 corners pre-populated (ONLY ON FIRST LOAD)
            # If we rely on session_state for restarts, we only pass initial_drawing once?
            # st_canvas uses 'initial_drawing' only if the canvas is empty/init.
            
            # We need to construct initial_drawing object list from session_state for CONSISTENCY
            # so that if the component remounts, it places points where we think they are.
            initial_objects = []
            for i, (cx, cy) in enumerate(st.session_state['calibration_corners']):
                initial_objects.append({
                    "type": "circle",
                    "left": cx - 8, # Adjust for radius to center
                    "top": cy - 8,
                    "radius": 8, # Increased from 5
                    "fill": "rgba(0, 255, 0, 0.5)",
                    "stroke": "#00FF00",
                    "strokeWidth": 1
                })
                
            initial_drawing = {
                "version": "4.4.0",
                "objects": initial_objects
            }

            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",
                stroke_width=1,
                stroke_color="#00FF00",
                background_image=Image.fromarray(bg_image) if bg_image is not None else None,
                update_streamlit=True,
                height=canvas_height,
                width=canvas_width,
                drawing_mode="transform", # Allow dragging
                initial_drawing=initial_drawing,
                key=f"calibration_canvas_{st.session_state.canvas_version}", # Dynamic key for reset
            )
            
            current_corners_canvas = []
            if canvas_result.json_data is not None:
                objects = canvas_result.json_data["objects"]
                # Filter for circles
                circles = [obj for obj in objects if obj["type"] == "circle"]
                
                # Check if we have 4 circles (sometimes users strictly add/delete, but here we assume drag)
                if len(circles) == 4:
                    # Sort them? No, we hope they stay in order 0,1,2,3...
                    # Wait, st_canvas order might change?
                    # Ideally we sort by position to ensure TL, TR, BR, BL?
                    # But the user might drag TL to BR...
                    # Let's trust the order of objects in the list if stable.
                    # Usually Fabric.js pushes to end on selection? That ruins order.
                    # We need a robust way to match circles to corners.
                    # Heuristic: Sort by Y then X?
                    # TL: Low Y, Low X. TR: Low Y, High X. BR: High Y, High X. BL: High Y, Low X.
                    pass 
                
                for obj in circles:
                    r = obj.get("radius", 5) * obj.get("scaleX", 1)
                    cx = int(obj["left"] + r)
                    cy = int(obj["top"] + r)
                    current_corners_canvas.append((cx, cy))

            # Logic to update lines
            if len(current_corners_canvas) == 4:
                # Basic sorting to ensure consistent box drawing
                # Sort by Y (Top vs Bottom)
                sorted_y = sorted(current_corners_canvas, key=lambda k: k[1])
                top_two = sorted(sorted_y[:2], key=lambda k: k[0]) # Left, Right
                bottom_two = sorted(sorted_y[2:], key=lambda k: k[0]) # Left, Right (Wait, BR is Right. BL is Left)
                # Order: TL, TR, BR, BL (to match polylines connection)
                # top_two[0] = TL, top_two[1] = TR
                # bottom_two[1] = BR, bottom_two[0] = BL (sorted by X: BL < BR)
                
                ordered_corners = [top_two[0], top_two[1], bottom_two[1], bottom_two[0]]
                
                # Check for change
                if ordered_corners != st.session_state['calibration_corners']:
                    st.session_state['calibration_corners'] = ordered_corners
                    st.rerun()
            
            # Map FINAL corners to Original Resolution for processing
            final_corners = []
            corners_to_map = st.session_state['calibration_corners']
            for cx, cy in corners_to_map:
                orig_x = int(cx * (width / canvas_width))
                orig_y = int(cy * (height / canvas_height))
                final_corners.append((orig_x, orig_y))
            
            st.write(f"Points Detected: {len(final_corners)} / 4")
            
            # Palette: Action buttons (Reset & Confirm)
            col_reset, col_confirm = st.columns([1, 2])

            with col_reset:
                if st.button("↺ Reset Corners", help="Reset corners to default positions", use_container_width=True):
                    # Reset to defaults (20%, 80% inset)
                    st.session_state['calibration_corners'] = [
                        (int(canvas_width * 0.2), int(canvas_height * 0.2)), # TL
                        (int(canvas_width * 0.8), int(canvas_height * 0.2)), # TR
                        (int(canvas_width * 0.8), int(canvas_height * 0.8)), # BR
                        (int(canvas_width * 0.2), int(canvas_height * 0.8))  # BL
                    ]
                    st.session_state.canvas_version += 1
                    st.rerun()

            should_process = False
            with col_confirm:
                if len(final_corners) == 4:
                    # Only show confirm if we have 4 points
                    if st.button("✅ Confirm Calibration", type="primary", use_container_width=True):
                        st.session_state.corners = final_corners
                        should_process = True

            # Check if processing was confirmed (using flag or direct button if not nested)
            # We use a local variable `should_process` so it only runs on the click event interaction.

            if should_process:
                with st.status("Processing video...", expanded=True) as status:
                    st.write("🚀 Initializing tracking models...")

                    # --- PROCESSING PIPELINE (cached models) ---
                    tracker = load_yolo_tracker()
                    shuttle_tracker = load_tracknet_tracker()
                    geometry = GeometryEngine()

                    # Basic sorter to ensure TL, TR, BL, BR order roughly
                    # Sort by Y (Top vs Bottom), then by X (Left vs Right)
                    sorted_corners = sorted(final_corners, key=lambda p: p[1]) # Top 2, Bottom 2
                    top_corners = sorted(sorted_corners[:2], key=lambda p: p[0]) # TL, TR
                    bottom_corners = sorted(sorted_corners[2:], key=lambda p: p[0]) # BL, BR

                    # Order: TL, TR, BL, BR for GeometryEngine logic (check implementation?)
                    # Standard usually TL, TR, BR, BL or TL, TR, BL, BR.
                    # Let's assume TL, TR, BL, BR based on previous slider code logic
                    final_input_corners = [top_corners[0], top_corners[1], bottom_corners[0], bottom_corners[1]]

                    geometry.calculate_homography(final_input_corners)
                    analysis = AnalysisEngine(fps=info['fps'])

                    # Create a temporary output file
                    tfile_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')

                    # Open video with GPU decoding if available
                    cap = GPUVideoReader(video_path, use_gpu=True)

                    # Seek to start frame
                    if start_frame_idx > 0:
                        cap.seek(start_frame_idx)

                    # Setup Async Output Writer with H.264 codec (no transcoding needed)
                    out = create_h264_writer(tfile_out.name, info['fps'], (width, height), queue_size=64)

                    frame_idx = start_frame_idx
                    max_frames = info['frame_count']

                    progress_bar = st.progress(0)

                    # Data Collection
                    # --- PARALLEL PROCESSING PIPELINE ---
                    st.write("⚡ Running TrackNet + YOLO in parallel...")

                    player_history, shuttle_history, player_present_frames, camera_cut_frames = \
                        process_video_concurrent(
                            video_path=video_path,
                            tracker=tracker,
                            shuttle_tracker=shuttle_tracker,
                            geometry=geometry,
                            output_writer=out,
                            start_frame=start_frame_idx,
                            max_frames=max_frames,
                            progress_bar=progress_bar,
                            batch_size=16  # Increased from 8 for better GPU utilization
                        )

                    status.update(label="✅ Processing Complete!", state="complete", expanded=False)
                    
                    st.success(f"✅ Processing complete: {len(shuttle_history)} shuttle detections, {len(player_history)} player frames")
                    
                    out.release()
                    
                    # Check if transcoding is needed (H.264 written directly skips this)
                    # Use ffprobe to check codec
                    probe_cmd = [
                        "ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=codec_name",
                        "-of", "default=noprint_wrappers=1:nokey=1",
                        tfile_out.name
                    ]
                    try:
                        codec = subprocess.check_output(probe_cmd, stderr=subprocess.DEVNULL).decode().strip()
                    except:
                        codec = "unknown"
                    
                    if codec == "h264":
                        # Already H.264, no transcoding needed!
                        st.success("✅ Video already H.264 - no transcoding needed!")
                        converted_file = tfile_out.name
                    else:
                        # Need to transcode (mp4v fallback)
                        st.info(f"⚡ Quick transcoding ({codec} → H.264 NVENC)...")
                        converted_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name

                        cmd_nvenc = [
                            "ffmpeg", "-y", "-hwaccel", "cuda", "-i", tfile_out.name,
                            "-c:v", "h264_nvenc", "-preset", "p4", "-pix_fmt", "yuv420p", converted_file
                        ]
                        cmd_cpu = [
                            "ffmpeg", "-y", "-i", tfile_out.name,
                            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", converted_file
                        ]

                        result = subprocess.run(cmd_nvenc).returncode
                        if result != 0:
                            subprocess.run(cmd_cpu)

                        # Cleanup intermediate file
                        if os.path.exists(tfile_out.name):
                            try:
                                os.remove(tfile_out.name)
                            except Exception as e:
                                print(f"Warning: Failed to remove intermediate file: {e}")
                    
                    # Run Analysis Engine Here (Before saving state)
                    analysis_report = analysis.analyze_rally(
                        player_history, 
                        shuttle_history, 
                        camera_cut_frames,
                        player_present_frames,
                        shuttle_tracker,
                        geometry,
                        video_path=video_path
                    )

                    # CLEANUP OLD ARTIFACTS
                    if st.session_state.get('analysis_results'):
                        old_results = st.session_state['analysis_results']
                        if 'clips_dir' in old_results and os.path.exists(old_results['clips_dir']):
                            try:
                                shutil.rmtree(old_results['clips_dir'])
                            except Exception as e:
                                print(f"Warning: Failed to cleanup old clips dir: {e}")
                        # Also clean up old converted file if it exists and is different from current?
                        # No, we are creating a NEW one here, so the old one is garbage.
                        if 'converted_file' in old_results and os.path.exists(old_results['converted_file']):
                            try:
                                os.remove(old_results['converted_file'])
                            except Exception as e:
                                print(f"Warning: Failed to cleanup old converted file: {e}")

                    # Create NEW persistent clips_dir for this analysis session
                    clips_dir = tempfile.mkdtemp(prefix="rally_clips_")

                    st.session_state['analysis_results'] = {
                        'analysis_report': analysis_report,
                        'converted_file': converted_file,
                        'info': info,
                        'clips_dir': clips_dir
                    }
                    
                    # --- AUTO-RUN AI IF KEY PRESENT ---
                    if gemini_api_key:
                        with st.spinner("🤖 Auto-Generating Tactical Analysis with Gemini..."):
                            try:
                                coach = GeminiCoach(gemini_api_key)
                                insights = coach.analyze_match(analysis_report, focus_player="Near") # Default focus
                                st.session_state.coach_insights = insights
                                st.session_state.coach_focus = "Near"
                                st.session_state.coach_chat = []
                                # Palette: Flag completion for toast
                                st.session_state.processing_complete = True
                            except Exception as e:
                                # Palette: Pass error to next run
                                st.session_state.processing_error = str(e)
                                # We still count processing as complete because the video part worked
                                st.session_state.processing_complete = True
                    else:
                        # No AI key, but video processing is done
                        st.session_state.processing_complete = True

                    st.rerun()

    with col2:
        st.subheader("Video Info")
        # Palette: Improved Data Display using Metrics
        st.metric("Resolution", f"{info['width']}x{info['height']}")
        st.metric("FPS", f"{info['fps']:.2f}")
        st.metric("Duration", f"{info['frame_count']/info['fps']:.1f}s")
        
    # --- RESULTS (Rendered from Session State) ---
    if st.session_state.get('analysis_results'):
        st.divider()
        render_results(st.session_state['analysis_results'], info, gemini_api_key)

else:
    ui.stepper(0)
    st.markdown("### Turn match footage into coaching")
    c1, c2, c3 = st.columns(3)
    for col, (icon, title, body) in zip((c1, c2, c3), (
        ("📂", "Upload", "Add a singles match video (MP4, MOV, WebM, MKV) from the sidebar."),
        ("📐", "Calibrate", "Drag the 4 green points onto the inner singles court lines."),
        ("🧠", "Analyze", "YOLO pose + TrackNet shuttle + SlowFast shots, then Gemini coaching."),
    )):
        with col:
            ui.card(f"{icon} {title}: {body}")
    st.info("💡 Best results: fixed camera, behind the baseline or high rear angle.")
