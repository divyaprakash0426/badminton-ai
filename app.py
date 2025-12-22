import streamlit as st
import tempfile
import cv2
import numpy as np
import os
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

st.set_page_config(page_title="Badminton Singles Analytics", page_icon="🏸", layout="wide")

st.title("🏸 Badminton Singles Analytics Engine")

# --- session state ---
if 'corners' not in st.session_state:
    st.session_state.corners = None
if 'coach_insights' not in st.session_state:
    st.session_state.coach_insights = None

# --- sidebar ---
st.sidebar.header("Configuration")

# Check Device
import torch
device_name = "🟢 GPU (CUDA)" if torch.cuda.is_available() else "🟠 CPU"
st.sidebar.image("https://img.icons8.com/color/48/000000/processor.png", width=30)
st.sidebar.markdown(f"**Processor:** {device_name}")
if not torch.cuda.is_available():
    st.sidebar.caption("Use `supergfxctl` to enable RTX 4080 if available.")


# API Keys
st.sidebar.divider()
st.sidebar.markdown("### 🤖 AI Coach")
gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Required for Tab 3 AI features.")

uploaded_file = st.sidebar.file_uploader(
    "Upload Video",
    type=["mp4", "mov", "avi", "webm", "mkv"],
    help="Limit 200MB. Supported formats: MP4, MOV, AVI, WebM, MKV"
)

# --- main ---
if uploaded_file:
    # Save uploaded file to temp (Securely in chunks)
    video_path = save_uploaded_file_securely(uploaded_file)
    if not video_path:
        st.error("Error saving file. It might exceed the size limit.")
        st.stop()

    original_suffix = os.path.splitext(video_path)[1].lower()

    # Pre-transcode WebM/MKV files (often use AV1 codec not supported by OpenCV)
    if original_suffix in ['.webm', '.mkv']:
        st.info("Transcoding WebM/MKV to H.264 for compatibility...")
        transcoded_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        transcode_cmd = [
            "ffmpeg", "-y", "-i", video_path, "-c:v", "libx264", "-preset", "fast",
            "-crf", "23", "-c:a", "aac", transcoded_path
        ]
        result = subprocess.run(transcode_cmd, stderr=subprocess.DEVNULL).returncode
        if result == 0 and os.path.exists(transcoded_path) and os.path.getsize(transcoded_path) > 0:
            video_path = transcoded_path
            st.success("Transcoding complete!")
        else:
            st.error("Transcoding failed. Please try a different video format.")
    
    vp = VideoProcessor()
    info = vp.get_video_info(video_path)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("1. Court Calibration")
        
        # Frame Selector for Calibration
        st.markdown("**Select Calibration Frame** (Skip intros)")
        start_frame_idx = st.slider("Frame Index", 0, info['frame_count']-1, 0)
        
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
                    "left": cx - 5, # Adjust for radius to center
                    "top": cy - 5,
                    "radius": 5,
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
                key="calibration_canvas_v2", # New key to force reload if needed
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
            
            if len(final_corners) == 4:
                st.success("4 Corners Connected!")
                if st.button("✅ Confirm Calibration & Process", type="primary", use_container_width=True):
                    st.session_state.corners = final_corners
                    
                    # Palette: Use st.status for better feedback
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

                    st.session_state['analysis_results'] = {
                        'analysis_report': analysis_report,
                        'converted_file': converted_file,
                        'info': info
                    }
                    
                    # --- AUTO-RUN AI IF KEY PRESENT ---
                    if gemini_api_key:
                        with st.spinner("🤖 Auto-Generating Tactical Analysis with Gemini..."):
                            try:
                                coach = GeminiCoach(gemini_api_key)
                                insights = coach.analyze_match(analysis_report, focus_player="Near") # Default focus
                                st.session_state.coach_insights = insights
                                st.success("AI Analysis Ready!")
                            except Exception as e:
                                st.error(f"Auto-AI failed: {e}")
                    
                    st.rerun()

                    # --- RESULTS (Rendered from Session State) ---
                    if st.session_state.get('analysis_results'):
                        results = st.session_state['analysis_results']
                        analysis_report = results['analysis_report']
                        converted_file = results['converted_file']
                        # info = results['info']
                        
                        st.divider()
                        st.subheader("2. Analysis Results")
                        
                        tab1, tab2, tab3 = st.tabs(["🎥 Video Overlay", "📊 Tactical Analysis", "🤖 AI Coach"])
                        
                        with tab1:
                            st.video(converted_file)
                            
                        with tab2:
                            st.markdown("### The Coach's Corner (Gemini 3)")
                            
                            # Display Stats
                            st.metric("Total Distance Covered", f"{analysis_report['total_distance']:.2f} m")
                            st.metric("Max Speed", f"{analysis_report['max_speed']:.2f} m/s")
                            
                            st.subheader("Rally Analysis")
                            st.write(f"**Total Shots Detected:** {analysis_report.get('total_shots', 0)}")
                            
                            # Pre-calculate Feedback for UI
                            # Use default focus player "Near" for initial display, user can toggle in Tab 3 but Tab 2 is general
                            # Ideally, we should add a small toggle in Tab 2 or valid global state
                            # For now, let's assume User is "Near" (Blue) for the tactical tips
                            
                            coach_verdict_ui = analysis.generate_ai_verdict(analysis_report, focus_player="Near")
                            rally_feedback_map = {item['rally_index']: item for item in coach_verdict_ui['rally_feedback']}
                            
                            # Shot Distribution
                            if 'shot_distribution' in analysis_report:
                                st.write("#### Shot Distribution")
                                st.json(analysis_report['shot_distribution'])
                            
                            # Rally Breakdown with Video Clips
                            st.write("#### Detailed Rallies")
                            if 'rallies' in analysis_report:
                                clips_dir = tempfile.mkdtemp(prefix="rally_clips_")
                                
                                for i, rally in enumerate(analysis_report['rallies']):
                                    shot_count = rally.get('shot_count', len(rally.get('shots', [])))
                                    start_frame = rally.get('start_frame', 0)
                                    end_frame = rally.get('end_frame', 0)
                                    
                                    # Calculate time range
                                    start_time = start_frame / info['fps']
                                    duration = (end_frame - start_frame) / info['fps']
                                    
                                    with st.expander(f"🏸 Rally {i+1} ({shot_count} shots) | Frames {start_frame}-{end_frame}"):
                                        
                                        # --- TACTICAL ADVICE (Updated UI) ---
                                        # Show this AT THE TOP
                                        
                                        # 1. Check for LLM Insights (High Priority)
                                        llm_critique = None
                                        if st.session_state.coach_insights:
                                            critiques = st.session_state.coach_insights.get('rally_critiques', {})
                                            llm_critique = critiques.get(str(i))
                                        
                                        if llm_critique:
                                            # LLM Feedback
                                            st.info(f"🤖 **AI Coach**: {llm_critique}")
                                        else:
                                            # 2. Fallback to Rule-Based (Immediate)
                                            feedback_item = rally_feedback_map.get(i)
                                            if feedback_item:
                                                comment = feedback_item['comment']
                                                winner = feedback_item['winner']
                                                
                                                if winner == "Near":
                                                    st.success(f"**Tactical Win**: {comment}")
                                                elif winner == "Far":
                                                    st.warning(f"**Tactical Tip**: {comment}")
                                                else:
                                                    st.info(f"**Insight**: {comment}")
                                            
                                            # Prompt user to generate AI insights if not present
                                            if not st.session_state.coach_insights and i == 0:
                                                st.caption("💡 Go to Tab 3 and click 'Generate Coach Analysis' for deeper AI insights on every rally!")
                                        
                                        # Extract clip using ffmpeg
                                        clip_path = os.path.join(clips_dir, f"rally_{i+1}.mp4")
                                        
                                        # Only extract if not already done
                                        if not os.path.exists(clip_path):
                                            ffmpeg_extract = [
                                                "ffmpeg", "-y", "-ss", f"{start_time:.2f}", "-i", converted_file,
                                                "-t", f"{duration:.2f}", "-c", "copy", clip_path
                                            ]
                                            subprocess.run(ffmpeg_extract, stderr=subprocess.DEVNULL)
                                        
                                        # Display clip if it exists
                                        if os.path.exists(clip_path) and os.path.getsize(clip_path) > 0:
                                            st.video(clip_path)
                                        else:
                                            st.warning("Clip extraction failed for this rally.")
                                            
                                        # Show shots breakdown with direction and player
                                        st.write("**Shot-by-Shot Breakdown:**")
                                        shots = rally.get('shots', rally if isinstance(rally, list) else [])
                                        for shot in shots:
                                            hit_by = shot.get('hit_by', '?')
                                            full_type = shot.get('full_type', shot['type'])
                                            player_emoji = "🔵" if hit_by == "Near" else "🟢"
                                            st.write(f"{player_emoji} **{hit_by}**: {full_type} (Frames: {shot['from_frame']}-{shot['to_frame']})")
        
                        with tab3:
                            st.subheader("3. AI Coach Verdict 🤖 (Gemini 3.0)")
                            
                            if not gemini_api_key:
                                st.warning("Please enter your Gemini API Key in the sidebar to unlock this feature.")
                                st.info("The AI Coach analyzes your match statistics to derive high-level insights.")
                            else:
                                # Player Selection - Now safe to toggle!
                                focus_player = st.radio("Who are you?", ["Near Player (Blue)", "Far Player (Green)"], index=0, horizontal=True)
                                focus_id = "Near" if "Near" in focus_player else "Far"
                                
                                # Show current insights if they exist (even from auto-run)
                                insights = st.session_state.coach_insights
                                
                                # Allow re-running manually if needed
                                if st.button("Regenerate Coach Analysis"):
                                    with st.spinner("Consulting the AI Coach..."):
                                        coach = GeminiCoach(gemini_api_key)
                                        insights = coach.analyze_match(analysis_report, focus_player=focus_id)
                                        st.session_state.coach_insights = insights 
                                        st.success("Analysis Updated!")
                                
                                if insights:
                                    col_weak, col_imp = st.columns(2)
                                    
                                    with col_weak:
                                        st.markdown("### 🎯 Opponent Weaknesses")
                                        if insights.get('opponent_weakness'):
                                            for item in insights['opponent_weakness']:
                                                st.write(f"- {item}")
                                        else:
                                            st.write("No specific weaknesses detected.")
                                            
                                    with col_imp:
                                        st.markdown("### 📈 My Improvements")
                                        if insights.get('my_improvements'):
                                            for item in insights['my_improvements']:
                                                st.write(f"- {item}")
                                        else:
                                            st.write("Keep up the good work!")

    with col2:
        st.subheader("Video Info")
        st.markdown(f"**Resolution:** {info['width']}x{info['height']}")
        st.markdown(f"**FPS:** {info['fps']:.2f}")
        st.markdown(f"**Duration:** {info['frame_count']/info['fps']:.1f}s")
        
    # --- RESULTS (Rendered from Session State) ---
    if st.session_state.get('analysis_results'):
        results = st.session_state['analysis_results']
        analysis_report = results['analysis_report']
        converted_file = results['converted_file']
        # info = results['info']
        
        st.divider()
        st.subheader("2. Analysis Results")
        
        tab1, tab2, tab3 = st.tabs(["🎥 Video Overlay", "📊 Tactical Analysis", "🤖 AI Coach"])
        
        with tab1:
            st.video(converted_file)
            
        with tab2:
            st.markdown("### The Coach's Corner (Gemini 3)")
            
            # Display Stats
            st.metric("Total Distance Covered", f"{analysis_report['total_distance']:.2f} m")
            st.metric("Max Speed", f"{analysis_report['max_speed']:.2f} m/s")
            
            st.subheader("Rally Analysis")
            st.write(f"**Total Shots Detected:** {analysis_report.get('total_shots', 0)}")
            
            # Pre-calculate Feedback for UI
            analysis = AnalysisEngine(fps=info['fps']) # Ensure instance is available during re-runs
            coach_verdict_ui = analysis.generate_ai_verdict(analysis_report, focus_player="Near")
            rally_feedback_map = {item['rally_index']: item for item in coach_verdict_ui['rally_feedback']}
            
            # Shot Distribution
            if 'shot_distribution' in analysis_report:
                st.write("#### Shot Distribution")
                st.json(analysis_report['shot_distribution'])
            
            # Rally Breakdown with Video Clips
            st.write("#### Detailed Rallies")
            if 'rallies' in analysis_report:
                clips_dir = tempfile.mkdtemp(prefix="rally_clips_")
                
                for i, rally in enumerate(analysis_report['rallies']):
                    shot_count = rally.get('shot_count', len(rally.get('shots', [])))
                    start_frame = rally.get('start_frame', 0)
                    end_frame = rally.get('end_frame', 0)
                    
                    # Calculate time range
                    start_time = start_frame / info['fps']
                    duration = (end_frame - start_frame) / info['fps']
                    
                    with st.expander(f"🏸 Rally {i+1} ({shot_count} shots) | Frames {start_frame}-{end_frame}"):
                        
                        # --- TACTICAL ADVICE ---
                        # 1. Check for LLM Insights (High Priority)
                        llm_critique = None
                        if st.session_state.coach_insights:
                            critiques = st.session_state.coach_insights.get('rally_critiques', {})
                            llm_critique = critiques.get(str(i))
                        
                        if llm_critique:
                            st.info(f"🤖 **AI Coach**: {llm_critique}")
                        else:
                            # 2. Fallback to Rule-Based
                            feedback_item = rally_feedback_map.get(i)
                            if feedback_item:
                                comment = feedback_item['comment']
                                winner = feedback_item['winner']
                                
                                if winner == "Near":
                                    st.success(f"**Tactical Win**: {comment}")
                                elif winner == "Far":
                                    st.warning(f"**Tactical Tip**: {comment}")
                                else:
                                    st.info(f"**Insight**: {comment}")
                            
                            # Prompt if missing
                            if not st.session_state.coach_insights and i == 0:
                                st.caption("💡 Go to Tab 3 to auto-generate deeper AI insights!")
                        
                        # Clip extraction (Lazy)
                        clip_path = os.path.join(clips_dir, f"rally_{i+1}.mp4")
                        if not os.path.exists(clip_path):
                            ffmpeg_extract = [
                                "ffmpeg", "-y", "-ss", f"{start_time:.2f}", "-i", converted_file,
                                "-t", f"{duration:.2f}", "-c", "copy", clip_path
                            ]
                            subprocess.run(ffmpeg_extract, stderr=subprocess.DEVNULL)
                        
                        if os.path.exists(clip_path) and os.path.getsize(clip_path) > 0:
                            st.video(clip_path)
                        else:
                            st.warning("Clip extraction failed for this rally.")
                            
                        # Breakdown
                        st.write("**Shot-by-Shot Breakdown:**")
                        shots = rally.get('shots', [])
                        for shot in shots:
                            hit_by = shot.get('hit_by', '?')
                            full_type = shot.get('full_type', shot['type'])
                            player_emoji = "🔵" if hit_by == "Near" else "🟢"
                            st.write(f"{player_emoji} **{hit_by}**: {full_type} (Frames: {shot['from_frame']}-{shot['to_frame']})")

        with tab3:
            st.subheader("3. AI Coach Verdict 🤖 (Gemini 3.0)")
            
            if not gemini_api_key:
                st.warning("Please enter your Gemini API Key in the sidebar.")
            else:
                # Player Selection - Now PERSISTENT
                focus_player = st.radio("Who are you?", ["Near Player (Blue)", "Far Player (Green)"], index=0, horizontal=True)
                focus_id = "Near" if "Near" in focus_player else "Far"
                
                curr_insights = st.session_state.coach_insights
                
                # Regenerate Button
                if st.button("Regenerate Coach Analysis"):
                    with st.spinner("Consulting the AI Coach..."):
                        coach = GeminiCoach(gemini_api_key)
                        curr_insights = coach.analyze_match(analysis_report, focus_player=focus_id)
                        st.session_state.coach_insights = curr_insights
                        st.success("Analysis Updated!")
                
                if curr_insights:
                    col_weak, col_imp = st.columns(2)
                    with col_weak:
                        st.markdown("### 🎯 Opponent Weaknesses")
                        for item in curr_insights.get('opponent_weakness', []):
                            st.write(f"- {item}")
                    with col_imp:
                        st.markdown("### 📈 My Improvements")
                        for item in curr_insights.get('my_improvements', []):
                            st.write(f"- {item}")

else:
    # 🎨 Palette: Empty State / Welcome Guide
    st.markdown("## 👋 Welcome to your AI Badminton Coach")

    st.markdown(
        """
        This tool analyzes your badminton singles matches to provide
        tactical insights and performance metrics using computer vision.
        """
    )

    with st.container(border=True):
        st.subheader("🚀 How to get started")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("### 1. Upload 📂")
            st.write("Use the sidebar to upload a match video (MP4, MOV, etc.).")

        with c2:
            st.markdown("### 2. Calibrate 📐")
            st.write("Drag the 4 green corners to match the singles court lines.")

        with c3:
            st.markdown("### 3. Analyze 🧠")
            st.write("Let the AI track players & shuttle to generate tactical advice.")

    st.info("💡 **Pro Tip:** For best results, use a video with a fixed camera angle from the back of the court.")
