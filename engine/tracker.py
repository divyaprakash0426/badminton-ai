import cv2
import numpy as np
import os
import torch
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor

# Try to import TrackNet dependencies, handle failure gracefully
import sys
# Add TrackNetV3 to path so its internal imports work (e.g. 'from model import ...')
tracknet_root = os.path.join(os.getcwd(), 'engine', 'tracknet_v3')
if tracknet_root not in sys.path:
    sys.path.append(tracknet_root)

try:
    # Now we can import directly as if we were in the tracknet folder
    from model import TrackNet
    from utils.general import get_model, HEIGHT, WIDTH, to_img
    TRACKNET_AVAILABLE = True
except ImportError as e:
    # Fallback/Debug
    print(f"TrackNet import failed: {e}")
    try:
        from engine.tracknet_v3.model import TrackNet
        from engine.tracknet_v3.utils.general import get_model, HEIGHT, WIDTH, to_img
        TRACKNET_AVAILABLE = True
    except ImportError as e2:
        TRACKNET_AVAILABLE = False
        print(f"TrackNet modules not found: {e2}. TrackNetTracker will be disabled.")

def predict_location(heatmap):
    """ Get coordinates from the heatmap (Ported from TrackNetV3/test.py). """
    # Optimization: Skip np.amax check and copy. heatmap should be uint8.
    (cnts, _) = cv2.findContours(heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not cnts:
        return 0, 0, 0, 0

    # Find max area rect efficiently
    max_area = 0
    best_rect = (0, 0, 0, 0)

    for ctr in cnts:
        rect = cv2.boundingRect(ctr)
        area = rect[2] * rect[3]
        if area > max_area:
            max_area = area
            best_rect = rect

    return best_rect


class YOLOTracker:
    def __init__(self, model_path='yolo11x-pose.pt', use_tensorrt=True):
        """
        Initialize the YOLOv11 Pose model with optional TensorRT acceleration.
        
        Args:
            model_path: Path to YOLO weights (.pt file)
            use_tensorrt: If True, export and use TensorRT engine on CUDA (2-4x faster)
        """
        import torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.is_tensorrt = False  # Track if using TensorRT (static batch=1)
        
        # TensorRT optimization for RTX GPUs
        if use_tensorrt and self.device == 'cuda':
            engine_path = model_path.replace('.pt', '.engine')
            
            if os.path.exists(engine_path):
                # Load existing TensorRT engine
                print(f"Loading TensorRT engine: {engine_path}")
                self.model = YOLO(engine_path)
                self.is_tensorrt = True
            else:
                # Export to TensorRT (one-time)
                print(f"Exporting {model_path} to TensorRT (one-time, may take 2-5 minutes)...")
                try:
                    base_model = YOLO(model_path)
                    base_model.export(format='engine', imgsz=960, half=True, device=0)
                    self.model = YOLO(engine_path)
                    self.is_tensorrt = True
                    print(f"TensorRT export complete: {engine_path}")
                except Exception as e:
                    print(f"TensorRT export failed: {e}. Using PyTorch model.")
                    self.model = YOLO(model_path)
        else:
            self.model = YOLO(model_path)

    def detect_pose(self, frame):
        """
        Detects pose in a single frame.
        Returns multiple keypoints: (N, 17, 2)
        """
        # Run inference at 960 resolution (balanced speed vs accuracy for far players)
        results = self.model(frame, verbose=False, imgsz=960)
        if results and results[0].keypoints is not None:
             # Check if any detections exist
             if results[0].keypoints.xy.shape[0] > 0:
                return results[0].keypoints.xy.cpu().numpy() # Shape (N, 17, 2)
        return []
    
    def detect_pose_batch(self, frames):
        """
        GPU-BATCHED pose detection for multiple frames.
        Note: TensorRT engines are static batch=1, so falls back to per-frame for .engine files.
        
        Args:
            frames: List of frames (H, W, 3) BGR format
            
        Returns:
            List of keypoints arrays, one per frame. Each is (N, 17, 2) or empty list.
        """
        if not frames:
            return []
        
        # TensorRT engines are built with static batch=1, must process individually
        if self.is_tensorrt:
            batch_keypoints = []
            for frame in frames:
                kp = self.detect_pose(frame)
                batch_keypoints.append(kp)
            return batch_keypoints
        
        # PyTorch model supports batch inference directly
        results = self.model(frames, verbose=False, imgsz=960)
        
        batch_keypoints = []
        for result in results:
            if result.keypoints is not None and result.keypoints.xy.shape[0] > 0:
                batch_keypoints.append(result.keypoints.xy.cpu().numpy())
            else:
                batch_keypoints.append([])
        
        return batch_keypoints

    def filter_and_sort_players(self, candidates, play_area_polygon, geometry_engine):
        """
        Filters candidates based on Play Area and sorts them (Near/Far).
        
        candidates: List of keypoint arrays (N, 17, 2)
        play_area_polygon: List of (x,y) tuples defining the valid court area + margin
        geometry_engine: Instance of GeometryEngine for 'is_in_play_area' check
        
        Returns:
            sorted_players: List of keypoints [Near_Player, Far_Player] (or partial list)
        """
        valid_players = []
        
        for kp in candidates:
            # Check ankles (15, 16)
            left_ankle = kp[15]
            right_ankle = kp[16]
            
            # Use midpoint of ankles as the "position" of the player
            if left_ankle[0] != 0 and right_ankle[0] != 0:
                pos = ((left_ankle[0] + right_ankle[0]) / 2, (left_ankle[1] + right_ankle[1]) / 2)
            elif left_ankle[0] != 0:
                pos = tuple(left_ankle)
            elif right_ankle[0] != 0:
                pos = tuple(right_ankle)
            else:
                # No ankles? Maybe use knees or center of bbox? Skip for now.
                continue
                
            # Filter: Check if inside Play Area
            if geometry_engine.is_in_play_area(pos, play_area_polygon):
                valid_players.append({'kp': kp, 'y': pos[1]})
                
        # Sort by Y (descending: larger Y is lower on screen -> "Near")
        # Near Player (Bottom) -> Index 0
        # Far Player (Top) -> Index 1 (if exists)
        valid_players.sort(key=lambda x: x['y'], reverse=True)
        
        return [p['kp'] for p in valid_players]

    def get_ankle_coordinates(self, keypoints):
        """
        Extracts ankle coordinates from keypoints.
        COCO keypoints: 15 (left ankle), 16 (right ankle)
        """
        if keypoints is None:
            return None, None
        
        # Ensure we have enough keypoints
        if len(keypoints) > 16:
            left_ankle = keypoints[15]
            right_ankle = keypoints[16]
            return left_ankle, right_ankle
        return None, None



class TrackNetTracker:
    def __init__(self, weights_path='engine/tracknet_v3/ckpts/TrackNet_best.pt',
                 inpaint_weights_path='engine/tracknet_v3/ckpts/InpaintNet_best.pt'):
        self.model = None
        self.inpaint_model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.seq_len = 3
        self.inpaint_seq_len = 16  # Default from TrackNetV3
        self.buffer = []
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        if not TRACKNET_AVAILABLE:
            print("TrackNet dependencies missing.")
            return

        # Load TrackNet
        if os.path.exists(weights_path):
            try:
                print(f"Loading TrackNet from {weights_path}...")
                checkpoint = torch.load(weights_path, map_location=self.device)
                param_dict = checkpoint['param_dict']
                self.seq_len = param_dict['seq_len']
                self.bg_mode = param_dict['bg_mode']
                
                self.model = get_model('TrackNet', self.seq_len, self.bg_mode).to(self.device)
                self.model.load_state_dict(checkpoint['model'])
                self.model.eval()
                
                # Enable FP16 for RTX 4080 (1.5-2x speedup)
                if self.device == 'cuda':
                    self.model = self.model.half()
                    self.use_fp16 = True
                    
                    self.use_fp16 = True
                    # Disable torch.compile to avoid CUDAGraphs stability issues on some setups
                    # self.model = torch.compile(self.model, mode='default')
                    print(f"TrackNet loaded successfully! (bg_mode: {self.bg_mode}, FP16: enabled, torch.compile: disabled)")
                else:
                    self.use_fp16 = False
                    print(f"TrackNet loaded successfully! (bg_mode: {self.bg_mode}, FP16: disabled)")
                
                if self.bg_mode == 'concat':
                    print("WARNING: Model uses 'concat' bg_mode. Median frame required but not implemented in streaming mode yet.")
            except Exception as e:
                print(f"Error loading TrackNet: {e}")
                self.model = None
        else:
            print(f"TrackNet weights not found at {weights_path}")
        
        # Load InpaintNet
        if os.path.exists(inpaint_weights_path):
            try:
                print(f"Loading InpaintNet from {inpaint_weights_path}...")
                inpaint_ckpt = torch.load(inpaint_weights_path, map_location=self.device)
                self.inpaint_seq_len = inpaint_ckpt['param_dict']['seq_len']
                self.inpaint_model = get_model('InpaintNet').to(self.device)
                self.inpaint_model.load_state_dict(inpaint_ckpt['model'])
                self.inpaint_model.eval()
                
                # Enable FP16 + torch.compile for InpaintNet (matching TrackNet optimizations)
                if self.device == 'cuda':
                    self.inpaint_model = self.inpaint_model.half()
                    # Disable torch.compile for stability
                    print(f"InpaintNet loaded successfully! (seq_len: {self.inpaint_seq_len}, FP16: enabled, torch.compile: disabled)")
                else:
                    print(f"InpaintNet loaded successfully! (seq_len: {self.inpaint_seq_len})")
            except Exception as e:
                print(f"Error loading InpaintNet: {e}")
                self.inpaint_model = None
        else:
            print(f"InpaintNet weights not found at {inpaint_weights_path}")
    
    def preprocess_frame(self, frame):
        """
        Preprocess a single frame for TrackNet.
        Returns: preprocessed_tensor
        """
        resized = cv2.resize(frame, (WIDTH, HEIGHT))
        img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0
        img_t = np.transpose(img_norm, (2, 0, 1))  # (3, H, W)
        return img_t
    
    def preprocess_batch(self, frames):
        """
        Parallel preprocessing of multiple frames using thread pool.
        
        CPU-bound operations (resize, color convert, normalize) are parallelized
        across 4 threads for ~2-3x speedup on multi-core CPUs.
        
        Args:
            frames: List of frames (H, W, 3) BGR format
            
        Returns:
            List of preprocessed tensors (3, H, W)
        """
        if not frames:
            return []
        
        # Use persistent executor
        return list(self.executor.map(self.preprocess_frame, frames))
    
    def track(self, frame):
        """
        Input: Raw frame (H_orig, W_orig, 3)
        Output: (x, y) or None
        """
        if self.model is None:
            return None
            
        h_orig, w_orig = frame.shape[:2]
        
        # 1. Resize and Preprocess
        resized = cv2.resize(frame, (WIDTH, HEIGHT))
        img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0
        img_t = np.transpose(img_norm, (2, 0, 1))
        
        # 2. Update Buffer
        self.buffer.append(img_t)
        if len(self.buffer) > self.seq_len:
            self.buffer.pop(0)
            
        # 3. Predict if buffer is full
        if len(self.buffer) == self.seq_len:
            input_tensor = np.concatenate(self.buffer, axis=0)
            
            if self.bg_mode == 'concat':
                median_dummy = self.buffer[0]
                input_tensor = np.concatenate((median_dummy, input_tensor), axis=0)

            input_tensor = torch.from_numpy(input_tensor).float().unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                y_pred = self.model(input_tensor)
                last_map = y_pred[0, -1, :, :].cpu().numpy()
                ret, last_map_binary = cv2.threshold(last_map, 0.5, 1, cv2.THRESH_BINARY)
                # Optimization: Pass binary image (0/1) directly to avoid 255 mult
                last_map_int = last_map_binary.astype(np.uint8)
                
                x, y, w, h = predict_location(last_map_int)
                
                if x == 0 and y == 0 and w == 0 and h == 0:
                     return None
                
                cx = x + w / 2
                cy = y + h / 2
                
                orig_x = int(cx * (w_orig / WIDTH))
                orig_y = int(cy * (h_orig / HEIGHT))
                
                return (orig_x, orig_y)
                
        return None
    
    def track_batch(self, frames):
        """
        GPU-BATCHED shuttle tracking for multiple frames.
        
        Args:
            frames: List of frames (H, W, 3) BGR format
            
        Returns:
            List of (x, y) or None for each frame
        """
        if self.model is None or len(frames) == 0:
            return [None] * len(frames)
        
        h_orig, w_orig = frames[0].shape[:2]
        results = [None] * len(frames)
        
        # 1. Preprocess all frames in parallel (CPU-bound parallelization)
        preprocessed = self.preprocess_batch(frames)
        
        # 2. Build batched input sequences
        # Each input needs seq_len consecutive frames
        batch_inputs = []
        batch_indices = []  # Track which output index each batch item corresponds to
        
        # Initialize with existing buffer
        extended_frames = list(self.buffer) + preprocessed
        
        for i in range(len(preprocessed)):
            # Window starts at buffer position + i
            start_idx = i
            end_idx = start_idx + self.seq_len
            
            if end_idx <= len(extended_frames):
                window = extended_frames[start_idx:end_idx]
                input_tensor = np.concatenate(window, axis=0)
                
                if self.bg_mode == 'concat':
                    median_dummy = window[0]
                    input_tensor = np.concatenate((median_dummy, input_tensor), axis=0)
                
                batch_inputs.append(input_tensor)
                batch_indices.append(i)
        
        # Update buffer with last seq_len-1 frames for next batch
        self.buffer = preprocessed[-(self.seq_len - 1):] if len(preprocessed) >= self.seq_len - 1 else \
                      (list(self.buffer) + preprocessed)[-(self.seq_len - 1):]
        
        if not batch_inputs:
            return results
        
        # 3. Stack into batch tensor and run inference (FP16 if available)
        batch_tensor = torch.from_numpy(np.stack(batch_inputs)).to(self.device)
        if getattr(self, 'use_fp16', False):
            batch_tensor = batch_tensor.half()
        else:
            batch_tensor = batch_tensor.float()
        
        with torch.no_grad():
            # torch.compile with 'reduce-overhead' handles CUDA Graphs internally
            y_pred = self.model(batch_tensor)  # (B, seq_len, H, W)
            
            # Process each prediction
            for batch_idx, output_idx in enumerate(batch_indices):
                # Convert to float32 for OpenCV compatibility (FP16 not supported by cv2.threshold)
                last_map = y_pred[batch_idx, -1, :, :].cpu().float().numpy()
                ret, last_map_binary = cv2.threshold(last_map, 0.5, 1, cv2.THRESH_BINARY)
                # Optimization: Pass binary image (0/1) directly to avoid 255 mult
                last_map_int = last_map_binary.astype(np.uint8)
                
                x, y, w, h = predict_location(last_map_int)
                
                if x == 0 and y == 0 and w == 0 and h == 0:
                    results[output_idx] = None
                else:
                    cx = x + w / 2
                    cy = y + h / 2
                    orig_x = int(cx * (w_orig / WIDTH))
                    orig_y = int(cy * (h_orig / HEIGHT))
                    results[output_idx] = (orig_x, orig_y)
        
        return results
    
    def track_video_batched(self, video_path, batch_size=16, start_frame=0, callback=None, progress_callback=None, total_frames=None):
        """
        Process entire video with GPU batching for maximum throughput.
        
        Args:
            video_path: Path to video file
            batch_size: Number of frames to process in each GPU batch (default 16)
            start_frame: Frame index to start processing from
            callback: Optional function(frame_idx, frame, shuttle_pos) called for each frame
            progress_callback: Optional function(current_frame, total_frames) for progress updates
            total_frames: Total frame count for progress calculation
            
        Returns:
            List of {'frame': int, 'pos': (x, y)} for detected shuttles
        """
        if self.model is None:
            print("TrackNet model not loaded.")
            return []
        
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Get total frames if not provided
        if total_frames is None:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        shuttle_history = []
        frame_idx = start_frame
        
        # Reset buffer for fresh video
        self.buffer = []
        
        while cap.isOpened():
            # Read batch of frames
            batch_frames = []
            batch_frame_indices = []
            
            for _ in range(batch_size):
                ret, frame = cap.read()
                if not ret:
                    break
                batch_frames.append(frame)
                batch_frame_indices.append(frame_idx)
                frame_idx += 1
            
            if not batch_frames:
                break
            
            # Process batch
            positions = self.track_batch(batch_frames)
            
            # Collect results and call callback
            for i, (f_idx, pos) in enumerate(zip(batch_frame_indices, positions)):
                if pos is not None:
                    shuttle_history.append({'frame': f_idx, 'pos': pos})
                
                if callback:
                    callback(f_idx, batch_frames[i], pos)
            
            # Update progress
            if progress_callback and total_frames > 0:
                progress_callback(frame_idx, total_frames)
        
        cap.release()
        print(f"GPU Batched tracking complete: {len(shuttle_history)} detections")
        return shuttle_history
    
    def rectify_trajectory(self, shuttle_history, img_w=1280, img_h=720):
        """
        Post-process trajectory using InpaintNet to fill gaps.
        
        Args:
            shuttle_history: List of {'frame': int, 'pos': (x, y)}
            img_w, img_h: Original image dimensions for normalization
            
        Returns:
            Enhanced shuttle_history with filled gaps
        """
        if self.inpaint_model is None or len(shuttle_history) < 20:
            return shuttle_history
        
        # 1. Build full frame range
        frames = [h['frame'] for h in shuttle_history]
        min_frame, max_frame = min(frames), max(frames)
        
        # Create lookup
        frame_to_pos = {h['frame']: h['pos'] for h in shuttle_history}
        
        # 2. Build coordinate array with visibility mask
        full_frames = list(range(min_frame, max_frame + 1))
        coords = []
        vis = []
        
        for f in full_frames:
            if f in frame_to_pos:
                pos = frame_to_pos[f]
                # Normalize to [0,1]
                coords.append([pos[0] / img_w, pos[1] / img_h])
                vis.append(1)
            else:
                coords.append([0.0, 0.0])
                vis.append(0)
        
        # 3. Generate inpaint mask (where to fill)
        vis_arr = np.array(vis)
        y_coords = np.array([c[1] for c in coords])
        th_h = 0.05  # Normalized threshold
        
        inpaint_mask = np.zeros_like(vis_arr)
        i = 0
        while i < len(vis_arr):
            # Find start of gap
            while i < len(vis_arr) - 1 and vis_arr[i] == 1:
                i += 1
            j = i
            # Find end of gap
            while j < len(vis_arr) - 1 and vis_arr[j] == 0:
                j += 1
            if j == i:
                break
            # Mark for inpainting if bounded by valid detections above threshold
            if i > 0 and j < len(vis_arr):
                if y_coords[i-1] > th_h and y_coords[j] > th_h:
                    inpaint_mask[i:j] = 1
            i = j
        
        # 4. Run InpaintNet in batches
        coords_tensor = torch.tensor(coords, dtype=torch.float32)
        mask_tensor = torch.tensor(inpaint_mask, dtype=torch.float32).unsqueeze(1)
        
        # Pad to match seq_len
        seq_len = self.inpaint_seq_len
        n_frames = len(coords_tensor)
        
        if n_frames < seq_len:
            # Pad with zeros
            pad_size = seq_len - n_frames
            coords_tensor = torch.cat([coords_tensor, torch.zeros(pad_size, 2)], dim=0)
            mask_tensor = torch.cat([mask_tensor, torch.zeros(pad_size, 1)], dim=0)
            n_frames = seq_len
        
        # Process in sliding windows
        inpainted_coords = coords_tensor.clone()
        
        with torch.no_grad():
            for start in range(0, n_frames - seq_len + 1, seq_len):
                end = start + seq_len
                coor_batch = coords_tensor[start:end].unsqueeze(0).to(self.device)
                mask_batch = mask_tensor[start:end].unsqueeze(0).to(self.device)
                
                # Use FP16 on CUDA (matching model precision)
                if self.device == 'cuda':
                    coor_batch = coor_batch.half()
                    mask_batch = mask_batch.half()
                
                c_inpaint = self.inpaint_model(coor_batch, mask_batch).float().cpu().squeeze(0)
                
                # Merge: use inpainted where mask is 1
                for i in range(seq_len):
                    if start + i < len(inpaint_mask) and inpaint_mask[start + i] == 1:
                        inpainted_coords[start + i] = c_inpaint[i]
        
        # 5. Build enhanced shuttle_history
        enhanced_history = []
        for i, f in enumerate(full_frames):
            if i < len(inpainted_coords):
                x = int(inpainted_coords[i][0].item() * img_w)
                y = int(inpainted_coords[i][1].item() * img_h)
                if x > 0 or y > 0:  # Only add if valid
                    enhanced_history.append({'frame': f, 'pos': (x, y)})
        
        print(f"InpaintNet: {len(shuttle_history)} -> {len(enhanced_history)} detections (filled {len(enhanced_history) - len(shuttle_history)} gaps)")
        return enhanced_history

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

