### 📂 Master Blueprint: Badminton Singles Analytics Engine (AI-Driven)

**Project Goal:** Build a "MVP" Python application to analyze single-player badminton videos.
**Input:** Single video file (mp4). Lateral or High-Back view.
**Output:** Streamlit Dashboard with annotated video, footwork heatmaps, and "Chunk" analysis.
**Target OS:** Linux/MacOS (Preferred for CV compatibility).

-----

### 1\. Technology Stack (Strict)

  * **Frontend:** `streamlit` (Simple, reactive UI).
  * **Core Logic:** Python 3.10+.
  * **Computer Vision:**
      * **Player Pose:** `ultralytics` (YOLO11-Pose). *Note: YOLO11 is the selected SOTA model.*
      * **Shuttle Tracking:** `TrackNetV3` (PyTorch implementation). *Task: Clone valid repo if V4 is unavailable.*
      * **Image Proc:** `opencv-python`, `numpy`, `pandas`.
  * **AI Coach:** `google-generativeai` (Gemini SDK) for tactical reasoning.
  * **Visualization:** `matplotlib` (for heatmaps), `supervision` (optional, for easy bounding box drawing).

-----

### 2\. Architecture: The "Singles Tactical Pipeline"

The application follows a linear data flow:

**Step 1: Ingestion & Homography**

  * User uploads video.
  * **UI Interaction:** User clicks 4 corners of the court on the first frame.
  * **System:** Calculates `cv2.getPerspectiveTransform` matrix to map pixels $\rightarrow$ meters.

**Step 2: The "Vision" Pass (Batch Processing)**

  * **Pass A (YOLO11):** Iterate video. Extract Player Keypoints (Ankles, Knees, Center of Mass).
  * **Pass B (TrackNet):** Extract Shuttle `(x, y)` coordinates. *Requires 3-frame stacking input.*

**Step 3: Event Construction ("The Chunking Engine")**

  * **Shot Detection:** Identify peaks in shuttle velocity/direction change.
  * **State Mapping:** For every shot, record:
      * `player_position` (Meters).
      * `opponent_position` (Inferred or tracked).
      * `base_position_error` (Distance from *Dynamic Base*, not Center).

**Step 4: The "Deep Think" (LLM)**

  * Send structured JSON of the rally to Gemini 3.
  * *Prompt Strategy:* "Analyze this sequence. Did the player bias their base position correctly given the shot quality? Identify patterns."

-----

### 3\. Implementation Steps for Antigravity Agent

**Prompt for Agent:** "Agent, execute the following implementation plan phase by phase. Do not move to Phase 2 until Phase 1 is verified."

#### **Phase 1: Scaffolding & Environment**

  * Create `requirements.txt`: `ultralytics`, `streamlit`, `opencv-python`, `pandas`, `shapely`, `google-generativeai`.
  * Setup project structure:
    ```
    /badminton_ai
    ├── app.py              # Streamlit Entry point
    ├── engine/
    │   ├── tracker.py      # YOLO11 & TrackNet Wrappers
    │   ├── geometry.py     # Homography & Physics logic
    │   └── analysis.py     # Chunk detection logic
    └── utils/
        └── video.py        # Video I/O helpers
    ```

#### **Phase 2: The Vision Engine (Backend)**

  * **Task A:** Implement `tracker.py`. Load `yolo11n-pose.pt`. Create a function `get_pose_data(frame)` that returns ankle coordinates.
  * **Task B:** Implement `TrackNet`. *Critical:* Since TrackNet is complex to install, instruct the agent to "Write a mock shuttle tracker first using simple color thresholding or background subtraction to unblock the pipeline, then integrate TrackNet weights later." (This keeps you moving).

#### **Phase 3: Geometry & Logic**

  * **Task:** Implement `geometry.py`.
  * **Logic:**
    ```python
    def calculate_dynamic_base(shot_type, player_pos):
        # Singles Logic: If I hit to net, base is closer to front.
        if shot_type == 'net_drop':
             return Point(center_x, center_y - 1.0) # Shift front
        return Point(center_x, center_y)
    ```

#### **Phase 4: The Streamlit Dashboard**

  * **Layout:**
      * **Sidebar:** Video Uploader, "Process" Button.
      * **Main Column:**
          * `st.video` (The processed video with skeleton overlay).
          * `st.line_chart` (Player speed over time).
          * **"The Coach's Corner":** A Markdown section streaming Gemini's analysis.

-----

### 4\. Key Algorithms (Copy into Agent's Context)

**A. Homography Transformation (Pixels to Meters)**

```python
import cv2
import numpy as np

def get_trans_matrix(pixel_corners):
    # Standard BWF court: 13.4m x 6.1m (Singles width 5.18m)
    real_corners = np.float32([[0,0], [6.1,0], [0,13.4], [6.1,13.4]])
    return cv2.getPerspectiveTransform(pixel_corners, real_corners)
```

**B. The "Chunk" Detector Logic**

  * **Definition:** A chunk is a sequence of 3 events: `Preparation -> Shot -> Recovery`.
  * **Logic:**
      * `Preparation`: Time from opponent contact $\rightarrow$ Player Split Step.
      * `Shot`: Player contact with shuttle.
      * `Recovery`: Time from contact $\rightarrow$ Reaching Dynamic Base.

-----

### 5\. Prompt to trigger Google Antigravity

**Copy/Paste this into Antigravity:**

> "I am building a Badminton Singles Analysis app using Python and Streamlit.
>
> **The Goal:** Analyze video to grade footwork and tactical patterns ('chunks') using Computer Vision and Gemini 3.
>
> **The Stack:**
>
> 1.  **YOLO11** (via `ultralytics`) for Pose Estimation.
> 2.  **TrackNetV3** (or generic object tracking if easier for MVP) for Shuttle.
> 3.  **Streamlit** for the frontend.
> 4.  **Gemini 3 API** for the coaching feedback.
>
> **Your First Task:**
> Create the `app.py` and `engine/geometry.py` files.
> In `app.py`, build a Streamlit interface that allows me to upload a video, displays the first frame, and lets me click 4 points to define the court corners. Store these corners in session state. Use `cv2` to draw the selected points on the frame in real-time."