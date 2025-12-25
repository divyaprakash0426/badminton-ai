import pandas as pd
import numpy as np
from scipy.signal import find_peaks

# =============================================================================
# SHOT CLASSIFICATION DECISION TABLE
# =============================================================================
# Each rule maps (start_zone, end_zone, speed_class) -> shot_type
# Rules are evaluated in order; first match wins.
# Zones: "front" (within 2.5m of net), "mid" (2.5-4m), "back" (>4m from net)
# Speed: "slow" (<8 m/s), "medium" (8-15 m/s), "fast" (>15 m/s)
# =============================================================================

# SHOT_CLASSIFICATION_RULES REMOVED - Using AI Model
# SHOT_CLASSIFICATION_RULES = []


def normalize_shot_type(s_type):
    """
    Normalizes a heuristic/simplified shot type to the VideoBadminton 18-class schema.
    Used for compatibility when ML model is not active.
    """
    s = s_type.lower()
    if "smash" in s: return "14_Smash"
    if "clear" in s: return "12_Clear"
    if "drop" in s: return "05_Drop Shot"
    if "lift" in s: return "02_Lift"
    if "net" in s: return "06_Push Shot" # Fallback for net shot
    if "drive" in s: return "11_Defensive Drive"
    if "short serve" in s or (s == "serve" and "short" in s): return "00_Short Serve"
    if "long serve" in s or (s == "serve" and "long" in s): return "13_Long Serve"
    if "serve" in s: return "00_Short Serve" # Default
from engine.geometry import ZoneMapper


class ScoreBoard:
    def __init__(self):
        self.score_a = 0
        self.score_b = 0
        self.history = []
        
    def update(self, rally_end_reason, last_shot_by, last_shot_in):
        """
        Infer point winner.
        rally_end_reason: "Land" or "Net"
        last_shot_by: "Near" or "Far" (Player A vs B)
        last_shot_in: True (In) or False (Out)
        """
        
        winner = None
        reason = ""
        
        if rally_end_reason == "Net":
            # Last player hit into net -> Other player wins
            winner = "Far" if last_shot_by == "Near" else "Near"
            reason = "Net Error"
        elif rally_end_reason == "Land":
            if last_shot_in:
                # Landed IN -> Hitter wins
                winner = last_shot_by
                reason = "Clean Winner"
            else:
                # Landed OUT -> Opponent wins
                winner = "Far" if last_shot_by == "Near" else "Near"
                reason = "Out"
                
        if winner == "Near":
             self.score_a += 1
        elif winner == "Far":
             self.score_b += 1
             
        self.history.append({
            "score": f"{self.score_a}-{self.score_b}",
            "winner": winner,
            "reason": reason
        })
        
        return self.score_a, self.score_b, winner


class AnalysisEngine:
    def __init__(self, fps=25.0):
        self.fps = fps
        self.shot_classifier = None
        try:
            from engine.action_recognition import ShotClassifier
            self.shot_classifier = ShotClassifier() # Uses pretrained or default
            print("Action Recognition Model Loaded.")
        except Exception as e:
            print(f"Could not load Action Recognition Model: {e}")

        self.zone_mapper = ZoneMapper()
        self.scoreboard = ScoreBoard()
        
    def calculate_distance(self, p1, p2):
        """Euclidean distance between two (x,y) points."""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def analyze_rally(self, player_history, shuttle_history=None, camera_cut_frames=None, player_present_frames=None, shuttle_tracker=None, geometry=None, video_path=None):
        """
        Analyzes the match using player and shuttle data.
        Returns a rich report dict including Match Story.
        """
        # 0. Reset Scoreboard per match analysis
        self.scoreboard = ScoreBoard() 
        
        # 1. Basic Player Movement Stats (Legacy)
        player_stats = self._analyze_player_movement(player_history)
        
        # 2. Shot & Rally Analysis (New)
        rally_stats = {"rallies": [], "total_shots": 0, "shot_distribution": {}, "match_story": []}
        if shuttle_history and len(shuttle_history) > 10:
             rally_stats = self._analyze_shots(
                 shuttle_history, 
                 camera_cut_frames or [],
                 player_present_frames or set(),
                 shuttle_tracker,
                 player_history,  # For player-based server detection
                 geometry,  # For court geometry direction
                 video_path
             )

        return {
            **player_stats,
            **rally_stats
        }

    def _analyze_player_movement(self, history):
        if not history:
             return {"total_distance": 0, "max_speed": 0, "chunks": []}
             
        # Vectorized NumPy approach (faster than pandas apply)
        positions = np.array([h['pos'] for h in history])
        
        if len(positions) < 2:
            return {"total_distance": 0, "max_speed": 0, "chunks": []}
        
        # Calculate step distances using vectorized operations
        diffs = np.diff(positions, axis=0)
        step_distances = np.sqrt(np.sum(diffs**2, axis=1))
        
        # Calculate speeds
        speeds = step_distances * self.fps
        
        # Smoothed speed using convolution (faster than rolling)
        if len(speeds) >= 5:
            kernel = np.ones(5) / 5
            speed_smooth = np.convolve(speeds, kernel, mode='valid')
            max_speed = float(np.max(speed_smooth)) if len(speed_smooth) > 0 else 0
        else:
            max_speed = float(np.max(speeds)) if len(speeds) > 0 else 0
        
        total_dist = float(np.sum(step_distances))
        
        return {
            "total_distance": total_dist,
            "max_speed": max_speed,
            "chunks": [] 
        }

    def _detect_swings(self, player_history):
        """
        Detects swing events based on high wrist velocity.
        Returns a dict of frame -> player_type ('Near'/'Far')
        """
        swings = {}
        if not player_history:
            return swings
        
        # Organize by player type
        p_data = {'Near': [], 'Far': []}
        for p in player_history:
            ptype = p.get('player_type', 'Near') # Default to Near
            p_data[ptype].append(p)
            
        for ptype, history in p_data.items():
            if len(history) < 3: continue
            
            df = pd.DataFrame(history)
            df = df.sort_values('frame')
            
            # Extract Wrist Coordinates
            # format: (x, y) tuple
            def calc_vel(col_name):
                # Calculate displacement magnitude between frames
                # Optimization: Vectorized coordinate extraction
                coords = pd.DataFrame(df[col_name].tolist(), columns=['x', 'y'], index=df.index)
                dx = coords['x'].diff()
                dy = coords['y'].diff()
                dist = np.sqrt(dx**2 + dy**2)
                dt = df['frame'].diff().fillna(1).replace(0, 1)
                return (dist / dt).fillna(0)

            l_vel = calc_vel('left_wrist')
            r_vel = calc_vel('right_wrist')
            
            # Max wrist velocity
            df['wrist_vel'] = np.maximum(l_vel, r_vel)
            
            # Threshold for swing (pixels/frame)
            # Fast flicks/smashes > 20, gentle shots > 10
            SWING_THRESH = 15 
            
            swing_frames = df[df['wrist_vel'] > SWING_THRESH]['frame'].tolist()
            for f in swing_frames:
                swings[int(f)] = ptype
                
        return swings

    def _analyze_shots(self, shuttle_history, camera_cut_frames=None, player_present_frames=None, shuttle_tracker=None, player_history=None, geometry=None, video_path=None):
        """
        Core logic to detect Hits -> Rallies -> Shot Types.
        Enhanced: Uses player positions and court geometry for accurate classification in Meters.
        """
        camera_cut_frames = camera_cut_frames or []
        player_present_frames = player_present_frames or set()
        player_history = player_history or []
        df = pd.DataFrame(shuttle_history)
        
        # --- 0. Filter Spurious Detections (Teleportation Filter) ---
        # Remove detections that appear as unrealistic jumps (false positives on player bodies)
        if len(df) > 2:
            # Optimization: Vectorized coordinate extraction (faster than apply)
            coords = pd.DataFrame(df['pos'].tolist(), columns=['x', 'y'], index=df.index)
            df['x'] = coords['x']
            df['y'] = coords['y']

            df['dx'] = df['x'].diff().abs()
            df['dy'] = df['y'].diff().abs()
            df['dist'] = np.sqrt(df['dx']**2 + df['dy']**2)
            df['frame_diff'] = df['frame'].diff().fillna(1).replace(0, 1)
            df['velocity'] = df['dist'] / df['frame_diff']
            
            # REBALANCED: 350px threshold (smashes can be ~300px/frame)
            MAX_VELOCITY = 350
            outlier_mask = df['velocity'] > MAX_VELOCITY
            
            # REBALANCED: Isolated spikes with gaps >8 frames
            df['prev_gap'] = df['frame'].diff() > 8
            df['next_gap'] = df['frame'].diff(-1).abs() > 8
            isolated_mask = df['prev_gap'] & df['next_gap']
            
            # Remove outliers and isolated spikes
            df = df[~outlier_mask & ~isolated_mask].copy()
            
            df = df.drop(columns=['x', 'y', 'dx', 'dy', 'dist', 'frame_diff', 'velocity', 
                                  'prev_gap', 'next_gap'], errors='ignore')
            
            if len(df) < 10:
                return {"rallies": [], "total_shots": 0, "shot_distribution": {}}
        
        # --- 1. Find Detection Gaps (Rally Boundaries) ---
        # When the shuttle lands, TrackNet loses it. Look for frame gaps > threshold.
        df['frame_gap'] = df['frame'].diff()
        
        # Threshold: If gap > 15 frames (~0.6s at 25fps), it's a new rally
        # (Lowered from 30 to catch more rally boundaries in broadcast videos)
        RALLY_GAP_FRAMES = 15
        
        # Find rally boundary indices (where gap exceeds threshold)
        rally_boundaries = df[df['frame_gap'] > RALLY_GAP_FRAMES].index.tolist()
        
        # --- NEW: Add camera cut frames as additional rally boundaries ---
        if camera_cut_frames:
            for cut_frame in camera_cut_frames:
                # Find the closest index in df to this cut frame
                closest_idx = (df['frame'] - cut_frame).abs().idxmin()
                if closest_idx not in rally_boundaries:
                    rally_boundaries.append(closest_idx)
            rally_boundaries = sorted(set(rally_boundaries))
        
        # Create rally segments
        rally_dfs = []
        start_idx = 0
        for boundary in rally_boundaries:
            rally_dfs.append(df.iloc[start_idx:boundary])
            start_idx = boundary
        rally_dfs.append(df.iloc[start_idx:]) # Last segment
        
        # Filter out very short segments (noise)
        # Must have >20 detection points AND span >1.5 seconds (37 frames at 25fps)
        MIN_RALLY_FRAMES = 37  # 1.5 seconds
        rally_dfs = [r for r in rally_dfs if len(r) > 20 and (r['frame'].max() - r['frame'].min()) > MIN_RALLY_FRAMES]
        
        # --- 2. Process Each Rally ---
        from collections import defaultdict
        classified_rallies = []
        shot_counts = defaultdict(int)
        total_shots = 0
        match_story = []
        
        for rally_df in rally_dfs:
            if rally_df.empty:
                continue
            
            # --- START TRIM: Detect "Launch" Event ---
            # Filter out pre-serve noise/holding where shuttle matches player movement or jitters
            # Calculate preliminary velocity
            temp_df = rally_df.copy()
            # Optimization: Vectorized coordinate extraction
            temp_coords = pd.DataFrame(temp_df['pos'].tolist(), columns=['x', 'y'], index=temp_df.index)
            temp_df['dx'] = temp_coords['x'].diff()
            temp_df['dy'] = temp_coords['y'].diff()
            temp_df['speed'] = np.sqrt(temp_df['dx']**2 + temp_df['dy']**2)
            
            # Find first "significant movement"
            # Threshold: > 15 pixels/frame (approx 0.5-1.0 meter/sec depending on scale)
            # Must be sustained for 2 frames
            LAUNCH_THRESH_PX = 15 
            
            valid_start_idx = -1
            speeds = temp_df['speed'].fillna(0).tolist()
            for i in range(len(speeds) - 2):
                if speeds[i] > LAUNCH_THRESH_PX and speeds[i+1] > LAUNCH_THRESH_PX:
                    valid_start_idx = i
                    break
            
            if valid_start_idx == -1:
                # No launch found (shuttle just hovered?)
                continue
                
            # Trim the rally to start from the launch
            # valid_start_idx corresponds to iloc of the df
            rally_df = rally_df.iloc[valid_start_idx:].copy()
            
            if len(rally_df) < 5: 
                continue

            # --- Validate player AND shuttle presence ---
            rally_start = int(rally_df['frame'].min())
            rally_end = int(rally_df['frame'].max())
            rally_duration_frames = rally_end - rally_start + 1
            frames_in_rally = set(range(rally_start, rally_end + 1))
            
            # Check player presence (at least 30%)
            player_frames = frames_in_rally & player_present_frames
            player_presence_ratio = len(player_frames) / rally_duration_frames if rally_duration_frames else 0
            
            # Check shuttle detection density (at least 50%)
            # rally_df contains actual shuttle detections in this segment
            # Allow lower density if using InpaintNet (which fills gaps later)
            shuttle_detection_ratio = len(rally_df) / rally_duration_frames if rally_duration_frames else 0
            
            # Both conditions must be met
            if player_presence_ratio < 0.3 or shuttle_detection_ratio < 0.3: # Relaxed shuttle check slightly
                continue  # Skip non-gameplay segments (animations, outros)
            
            # --- Apply InpaintNet PER-RALLY (fill gaps within this rally only) ---
            if shuttle_tracker is not None and hasattr(shuttle_tracker, 'rectify_trajectory') and shuttle_tracker.inpaint_model is not None:
                # Convert rally_df back to history format
                rally_history = [{'frame': int(row['frame']), 'pos': row['pos']} for _, row in rally_df.iterrows()]
                # Rectify this rally's trajectory
                rally_history = shuttle_tracker.rectify_trajectory(rally_history, img_w=1280, img_h=720)
                # Rebuild dataframe
                rally_df = pd.DataFrame(rally_history)
                if rally_df.empty or len(rally_df) < 5:
                    continue
                
            # Calculate velocity & direction for this rally
            rally_df = rally_df.copy()
            # Optimization: Vectorized coordinate extraction
            rally_coords = pd.DataFrame(rally_df['pos'].tolist(), columns=['x', 'y'], index=rally_df.index)
            rally_df['dx'] = rally_coords['x'].diff()
            rally_df['dy'] = rally_coords['y'].diff()
            rally_df['dt'] = rally_df['frame'].diff() / self.fps
            rally_df = rally_df.dropna()
            
            if rally_df.empty:
                continue
            
            # Angle of trajectory (radians)
            rally_df['angle'] = np.arctan2(rally_df['dy'], rally_df['dx'])
            rally_df['d_angle'] = rally_df['angle'].diff().abs()
            # Fix wrap-around: values > pi should be 2*pi - value
            rally_df['d_angle'] = rally_df['d_angle'].apply(lambda x: min(x, 2*np.pi - x) if pd.notna(x) else x)
            
            # Velocity Magnitude
            rally_df['speed_px'] = np.sqrt(rally_df['dx']**2 + rally_df['dy']**2) / rally_df['dt'].replace(0, np.nan)
            rally_df = rally_df.dropna()
            
            # --- 3. Detect Hit Events within this Rally ---
            # Detect swings first
            swing_events = self._detect_swings(player_history)
            
            # Helper: Check for swing near frame
            def has_swing(f, window=5):
                for val in range(int(f)-window, int(f)+window+1):
                    if val in swing_events:
                        return True
                return False

            ANGLE_THRESH_STRONG = 1.2 # Strong hit
            ANGLE_THRESH_WEAK = 0.8   # Weak hit (needs swing confirmation)
            SPEED_THRESH = 10
            
            # Vectorized swing check
            rally_frames = rally_df['frame'].values
            swing_mask = [has_swing(f) for f in rally_frames]
            rally_df['has_swing'] = swing_mask
            
            # Hit Conditions (Change in trajectory)
            # Condition 1: Strong directional change (Smash/High clear)
            cond1 = (rally_df['d_angle'] > ANGLE_THRESH_STRONG) & (rally_df['speed_px'] > SPEED_THRESH)
            
            # Condition 2: Medium directional change confirmed by wrist swing (Net shot/Drop)
            cond2 = (rally_df['d_angle'] > ANGLE_THRESH_WEAK) & (rally_df['speed_px'] > SPEED_THRESH) & (rally_df['has_swing'])
            
            potential_hits = rally_df[cond1 | cond2]
            
            # Group nearby frames to find single event (Clustering & Peak Detection)
            events = []
            
            # --- FORCE START EVENT ---
            first_row = rally_df.iloc[0]
            events.append({
                'frame': first_row['frame'],
                'pos': first_row['pos'],
                'is_forced_serve': True 
            })
            
            last_frame = first_row['frame']
            
            # 1. Cluster potential hits
            if not potential_hits.empty:
                hit_indices = potential_hits.index.tolist() # potential_hits is a slice of rally_df
                clusters = []
                if hit_indices:
                    current_cluster = [hit_indices[0]]
                    for i in range(1, len(hit_indices)):
                        # If index is adjacent or close in frame count
                        curr_idx = hit_indices[i]
                        prev_idx = hit_indices[i-1]
                        
                        curr_frame = rally_df.loc[curr_idx, 'frame']
                        prev_frame = rally_df.loc[prev_idx, 'frame']
                        
                        if curr_frame - prev_frame <= 5: # Cluster if within 5 frames
                            current_cluster.append(curr_idx)
                        else:
                            clusters.append(current_cluster)
                            current_cluster = [curr_idx]
                    clusters.append(current_cluster)
                
                # 2. Process clusters to find peaks
                MIN_INTER_SHOT_FRAMES = 12 # 0.5s debounce (prevents splitting Clears)
                
                for cluster in clusters:
                    # Get rows for this cluster
                    cluster_df = rally_df.loc[cluster]
                    
                    # Find Peak: Max directional change
                    peak_idx = cluster_df['d_angle'].idxmax()
                    peak_row = rally_df.loc[peak_idx]
                    
                    frame = peak_row['frame']
                    
                    # Debounce Check
                    if frame - last_frame > MIN_INTER_SHOT_FRAMES:
                        events.append({
                            'frame': frame,
                            'pos': peak_row['pos'],
                            'is_forced_serve': False
                        })
                        last_frame = frame
            
            if len(events) < 2:
                # Only served? Or just one hit detected? 
                # If we forced a serve, we have 1 event. If we found nothing else, reliable rally needs at least return?
                # Actually, an Ace serve is a valid rally of 1 shot. But usually we want analyze exchanges.
                # Let's keep it if > 1 (Serve + Result)
                if len(events) < 2:
                    continue 

                
            # --- 4. Classify Shots with Direction and Player Attribution ---
            rally_shots = []
            
            # Build player position lookup: frame -> {near_pos, far_pos}
            player_lookup = {}
            for p in player_history:
                frame = p['frame']
                if frame not in player_lookup:
                    player_lookup[frame] = {}
                # Store by player type (Near/Far)
                if p.get('player_type') == 'Near':
                    player_lookup[frame]['near'] = p['pos']
                elif p.get('player_type') == 'Far':
                    player_lookup[frame]['far'] = p['pos']
            
            # Court dimensions
            COURT_CENTER_X = 640  # Pixel fallback
            COURT_CENTER_Y = 360
            COURT_REAL_CENTER_X = 2.59  # Meters (5.18 / 2 for singles)
            
            # Helper: Find nearest player to shuttle with threshold
            def find_nearest_player(shuttle_pos, frame):
                nearest_frames = [f for f in player_lookup.keys() if abs(f - frame) <= 3]
                if not nearest_frames:
                    return None
                closest_frame = min(nearest_frames, key=lambda f: abs(f - frame))
                players = player_lookup.get(closest_frame, {})
                
                near_pos = players.get('near')
                far_pos = players.get('far')
                
                near_dist = self.calculate_distance(shuttle_pos, near_pos) if near_pos is not None else float('inf')
                far_dist = self.calculate_distance(shuttle_pos, far_pos) if far_pos is not None else float('inf')
                
                # Threshold: 3.0 meters (approx) or pixels? 
                # If using pixels, 100px is safe.
                # If logic is mixed (legacy pixels vs meters), let's use a safe pixel value.
                # Assuming 1280x720. 200px is safe reach.
                SAFE_DIST_PX = 200
                
                min_dist = min(near_dist, far_dist)
                if min_dist > SAFE_DIST_PX:
                    return None 
                
                if near_dist < far_dist:
                    return "Near"
                elif far_dist < near_dist:
                    return "Far"
                return None
            
            # Helper: Get direction using geometry if available
            def get_direction(start_pos, end_pos):
                # Get X coordinates
                start_x = start_pos[0]
                end_x = end_pos[0]
                
                if geometry and geometry.matrix is not None:
                    # Transform to real-world meters
                    try:
                        start_m = geometry.transform_point(start_pos)
                        end_m = geometry.transform_point(end_pos)
                        if start_m is not None and end_m is not None:
                            start_x = start_m[0]
                            end_x = end_m[0]
                            center_x = COURT_REAL_CENTER_X
                            threshold = 0.3  # 30cm from center is "Mid"
                        else:
                            center_x = COURT_CENTER_X
                            threshold = 80  # pixels
                    except:
                        center_x = COURT_CENTER_X
                        threshold = 80
                else:
                    center_x = COURT_CENTER_X
                    threshold = 80
                
                # Determine side relative to court center
                start_side = "Left" if start_x < center_x - threshold else ("Right" if start_x > center_x + threshold else "Mid")
                end_side = "Left" if end_x < center_x - threshold else ("Right" if end_x > center_x + threshold else "Mid")
                
                # Cross-court: opposites (Left->Right or Right->Left)
                if (start_side == "Left" and end_side == "Right") or (start_side == "Right" and end_side == "Left"):
                    return "Cross-Court"
                
                # Straight: same side (both Left or both Right)
                if start_side == end_side and start_side != "Mid":
                    return "Straight"
                
                # Mid shots are ambiguous - check X delta
                x_delta = abs(end_x - start_x)
                if geometry and geometry.matrix is not None:
                    # In meters
                    if x_delta > 1.5:  # More than 1.5m lateral movement
                        return "Cross-Court"
                    elif x_delta < 0.5:
                        return "Straight"
                else:
                    # In pixels
                    if x_delta > 150:
                        return "Cross-Court"
                    elif x_delta < 50:
                        return "Straight"
                
                return "Mid"
            
            for i in range(len(events) - 1):
                start = events[i]
                end = events[i+1]
                
                is_serve = (i == 0)
                
                # --- Shot Classification ---
                if is_serve:
                    shot_type = "Serve"
                    full_type = "Serve" # Placeholder, refined below
                else:
                    shot_type = self._classify_single_shot(start, end, rally_df, geometry, video_path)
                
                # Direction using geometry
                direction = get_direction(start['pos'], end['pos'])
                
                # --- Server / Player Attribution ---
                if is_serve:
                    # ZONE-BASED ATTRIBUTION (Primary)
                    # Use the starting Y position to determine who served.
                    # Top Half (Low Y) = Far Player
                    # Bottom Half (High Y) = Near Player
                    start_y = start['pos'][1]
                    
                    if start_y < COURT_CENTER_Y:
                        server = "Far"
                    else:
                        server = "Near"
                        
                    # Sanity Check with Trajectory (Secondary)
                    # If we detected "Far" zone but shuttle moves UP (dy < -50), something is wrong.
                    end_y = end['pos'][1]
                    dy = end_y - start_y
                    
                    if server == "Far" and dy < -50:
                        # Shuttle started Far but moved Farther? Or moved Up?
                        # dy negative = Up (towards Top/Far).
                        # If Far player serves, they hit it DOWN (dy > 0 usually).
                        # Unless it's a high clear serve.
                        pass
                    
                    # Overwrite only if STRICTLY CLOSE to other player
                    # (e.g. Near player intercepted a serve at the net? No, this is rally start)
                    
                    server_is_near = (server == "Near")
                
                # Alternate from the determined server
                if server_is_near:
                    hit_by = "Near" if i % 2 == 0 else "Far"
                else:
                    hit_by = "Far" if i % 2 == 0 else "Near"
                
                # Rule 0: Map AI Class names to readable names
                # The model returns e.g. "14_Smash". We strip the prefix.
                if "_" in shot_type and shot_type[0].isdigit():
                     clean_type = shot_type.split('_', 1)[1] # "14_Smash" -> "Smash"
                     shot_type = clean_type
                
                shot_counts[shot_type] += 1
                total_shots += 1


                
                # Full shot description
                if shot_type == "Serve":
                    # Classify serve type based on where it lands (Y coordinate)
                    end_y = end['pos'][1]
                    COURT_CENTER_Y = 360
                    # Far Serve: Lands near back boundary (Top or Bottom)
                    # Near Serve: Lands near service line (Middle)
                    
                    # If server is Near, they serve UP. Long serve lands small Y.
                    if server_is_near:
                        is_long = end_y < 200 # Top of screen
                    else:
                        is_long = end_y > 520 # Bottom of screen (720-200)
                        
                    serve_type = "Long Serve" if is_long else "Short Serve"
                    full_type = serve_type
                    shot_type = serve_type # Normalize the main type too
                else:
                    full_type = f"{direction} {shot_type}"
                
                # Calculate Tactical Zones
                start_zone = "Unknown"
                end_zone = "Unknown"
                start_m = None
                end_m = None
                
                if geometry and geometry.matrix is not None:
                     start_m = geometry.transform_point(start['pos'])
                     end_m = geometry.transform_point(end['pos'])
                     start_zone = self.zone_mapper.get_zone(start_m)
                     end_zone = self.zone_mapper.get_zone(end_m)
                
                rally_shots.append({
                    "from_frame": start['frame'],
                    "to_frame": end['frame'],
                    "type": shot_type,
                    "direction": direction,
                    "full_type": full_type,
                    "hit_by": hit_by,
                    "from_zone": start_zone,
                    "to_zone": end_zone,
                    "to_pos_m": end_m, 
                    "to_pos": end['pos']
                })
            
                if rally_shots:
                    # Add rally with frame range for clip extraction
                    # Use LAST HIT as end (not last shuttle detection) to avoid including next serve
                    first_hit_frame = int(events[0]['frame'])
                    last_hit_frame = int(events[-1]['frame'])
                    
                    # Add small buffer after last hit (1 second = 25 frames) to see point finish
                    BUFFER_FRAMES = 25
                    rally_end_with_buffer = min(last_hit_frame + BUFFER_FRAMES, int(rally_df['frame'].max()))
                    
                    # --- SCORING LOGIC ---
                    last_shot = rally_shots[-1]
                    last_hitter = last_shot['hit_by']
                    landing_pos = last_shot['to_pos'] if 'to_pos' in last_shot else None
                    
                    # Determine In/Out
                    is_in = True
                    is_net = False # TODO: Detect hitting net based on trajectory stop?
                    
                    # Simple geometry check for OUT
                    if geometry and last_shot['to_pos_m'] is not None:
                        lx_m, ly_m = last_shot['to_pos_m']
                        
                        # --- NEW: Get Relative Coordinates for AI Forecasting ---
                        if hasattr(geometry, 'get_relative_coordinates'):
                             rel_coords = geometry.get_relative_coordinates((lx_m, ly_m))
                             if rel_coords:
                                 last_shot['landing_x'], last_shot['landing_y'] = rel_coords
                             else:
                                 last_shot['landing_x'], last_shot['landing_y'] = None, None
                        
                        # Singles lines: x in [0.46, 5.64], y in [0, 13.4]
                        if lx_m < 0.46 or lx_m > 5.64 or ly_m < 0 or ly_m > 13.4:
                            is_in = False
                    
                    current_score_a, current_score_b, winner = self.scoreboard.update(
                        "Land" if not is_net else "Net", 
                        last_hitter, 
                        is_in
                    )
                    
                    rally_data = {
                        "shots": rally_shots,
                        "start_frame": first_hit_frame,
                        "end_frame": rally_end_with_buffer,
                        "shot_count": len(rally_shots),
                        "score_after": f"{current_score_a}-{current_score_b}",
                        "winner": winner,
                        # Add simple summary for CSV export if needed
                        "last_landing_x": last_shot.get('landing_x'),
                        "last_landing_y": last_shot.get('landing_y')
                    }
                    
                    classified_rallies.append(rally_data)
                    match_story.append(rally_data) # Add to story for LLM
        
        return {
            "rallies": classified_rallies,
            "total_shots": total_shots,
            "shot_distribution": shot_counts,
            "match_story": match_story
        }

    def _classify_single_shot(self, start_evt, end_evt, full_df, geometry=None, video_path=None):
        """
        Classifies a shot using purely the AI ShotClassifier.
        Falls back to "Unknown" if model fails or video not available.
        """
        
        # --- DL MODEL CLASSIFICATION ---
        if self.shot_classifier is not None and video_path is not None:
             try:
                 # Extract clip around the hit event (start_evt['frame'])
                 # We center the hit: frame - 8 to frame + 24 (total 32 frames)
                 # This captures a bit of preparation and the stroke execution/follow-through.
                 
                 import cv2
                 
                 center_frame = start_evt['frame']
                 start_f = max(0, center_frame - 8)
                 end_f = center_frame + 24 # 32 frames total
                 
                 cap = cv2.VideoCapture(video_path)
                 cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
                 
                 frames = []
                 for _ in range(end_f - start_f):
                     ret, frame = cap.read()
                     if not ret: break
                     # Convert BGR to RGB
                     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                     frames.append(frame)
                 cap.release()
                 
                 if len(frames) > 16: # Min requirement (half a clip)
                     # Convert to numpy (T, H, W, C)
                     vid_np = np.array(frames)
                     pred_class, conf = self.shot_classifier.predict(vid_np)
                     
                     # Return the raw class (e.g. "14_Smash")
                     # We trust the model even if low confidence, as heuristics are removed.
                     # But if really low, maybe "Unknown"?
                     if conf > 0.4: 
                        return pred_class
                     else:
                        print(f"Low confidence ({conf:.2f}) for shot at {center_frame}. Pred: {pred_class}")
                        return "Unknown"

             except Exception as e:
                 print(f"DL Classification failed: {e}")
                 
        return "Unknown"
                

    def generate_ai_verdict(self, rally_report, focus_player="Near"):
        """
        Generates tactical insights based on the rally analysis.
        focus_player: "Near" (Blue) or "Far" (Green) - The player we are coaching.
        """
        rallies = rally_report.get('rallies', [])
        if not rallies:
            return {
                "opponent_weakness": ["Not enough data to analyze opponent weaknesses."],
                "my_improvements": ["Play more rallies to get improvement suggestions."],
                "rally_feedback": []
            }
            
        opponent = "Far" if focus_player == "Near" else "Near"
        
        # 1. Gather Stats
        points_won = 0
        points_lost = 0
        winning_shots = [] # Shots I played that won the rally
        losing_causes = [] # Shots opponent played that won them the rally
        
        rally_feedback = []
        
        for rally in rallies:
            shots = rally.get('shots', [])
            if not shots:
                continue
                
            last_shot = shots[-1]
            # Heuristic: Who hit the last shot?
            # Usually, the person who hits the last shot is NOT the winner (unless it's a winner that lands IN).
            # But in computer vision without referee, we often lose tracking when the shuttle dies.
            # So:
            # Case A: I hit the last shot -> likely I made an error (net/out) OR I hit a winner.
            # Case B: Opponent hit the last shot -> likely they made an error OR they hit a winner.
            
            # Better Heuristic for CV (assuming valid gameplay):
            # If the rally ends, it's usually because the *receiving* player missed it or it went out.
            # Let's assume the last hitting player hitting a "Smash" or "Drive" likely won.
            # Let's assume the last hitting player hitting a "Net" or "Drop" that wasn't returned likely won.
            
            # We will use a simplified "Aggressor Wins" model for now:
            # If (Last Shot = Smash/Drive) and (No Return Detected) -> Hitter Won.
            # If (Last Shot = Lift/Clear) and (No Return Detected) -> Hitter probably hit OUT -> Receiver Won.
            
            last_hitter = last_shot['hit_by']
            shot_type = last_shot['type']
            
            winner = "?"
            # Aggressive shot that wasn't returned -> Hitter won
            if shot_type in ['Smash', 'Drive', 'Net', 'Drop']: 
                winner = last_hitter
            # Defensive/High shot that wasn't returned -> Likely Receiver won (Out)
            elif shot_type in ['Clear', 'Serve', 'Lift']: 
                winner = opponent if last_hitter == focus_player else focus_player
            
            # --- Per Rally Feedback ---
            comment = ""
            if winner == focus_player:
                points_won += 1
                winning_shots.append(last_shot)
                if shot_type == 'Smash':
                    comment = "🔥 **Great Smash!** That angle was unreachable. Keep attacking their backhand."
                elif shot_type == 'Net':
                    comment = "👌 **Nice Touch!** Excellent control taking it early at the net."
                elif shot_type == 'Drop':
                    comment = "✅ **Deceptive Drop!** You caught them moving backward."
                else:
                    comment = "👏 **Good Pressure!** You forced them into an error with consistent returns."
            else:
                points_lost += 1
                losing_causes.append(last_shot) # The shot that ended the rally
                
                # Coaching Advice based on HOW we lost
                if shot_type == 'Smash':
                    comment = "🛡️ **Lift Deeper!** Your lift was short, giving them an easy smash. Aim for the back tramlines."
                elif shot_type == 'Net':
                    comment = "⚠️ **Racket Up!** Be ready to pounce on net shots. Don't stand too far back."
                elif shot_type == 'Drop':
                    comment = "🦶 **Stay Mobile!** They caught you flat-footed. Keep your split step active."
                elif shot_type == 'Clear':
                    comment = "👀 **Judge it Better!** That clear was going out. Trust your judgment and let it fly."
                else:
                    comment = "🔄 **Reduce Errors!** Focus on getting the shuttle over the net safely rather than going for lines."
            
            rally_feedback.append({
                "rally_index": rallies.index(rally),
                "winner": winner,
                "comment": comment
            })

        # 2. Analyze Opponent Weakness (How did I win?)
        weakness_verdicts = []
        if winning_shots:
            df_win = pd.DataFrame(winning_shots)
            # Most effective winning shot
            if 'type' in df_win.columns:
                best_shot = df_win['type'].mode()
                if not best_shot.empty:
                    weakness_verdicts.append(f"**Vulnerable to {best_shot[0]}s**: You won {len(df_win[df_win['type'] == best_shot[0]])} points using this shot.")
            
            # Directional weakness
            if 'direction' in df_win.columns:
                best_dir = df_win['direction'].mode()
                if not best_dir.empty:
                    weakness_verdicts.append(f"**Weak on the {best_dir[0]}**: Opponent struggles to return {best_dir[0]} shots.")
        
        if not weakness_verdicts:
            weakness_verdicts.append("Keep playing to find opponent weaknesses.")

        # 3. Analyze My Improvements (How did I lose?)
        improvement_verdicts = []
        if losing_causes:
            df_loss = pd.DataFrame(losing_causes)
            # What kills me?
            if 'type' in df_loss.columns:
                fatal_shot = df_loss['type'].mode()
                if not fatal_shot.empty:
                    improvement_verdicts.append(f"**Defense against {fatal_shot[0]}**: You lost many points to opponent's {fatal_shot[0]}s.")
            
        # General Movement Stats
        # Check average speed calculation from earlier logic if available, or just general tip
        # For now, generic:
        if points_lost > points_won:
            improvement_verdicts.append("**Consistency**: Focus on keeping the shuttle in play rather than winners.")
        
        if not improvement_verdicts:
            improvement_verdicts.append("Solid play! Maintain your form.")
            
        return {
            "opponent_weakness": weakness_verdicts,
            "my_improvements": improvement_verdicts,
            "rally_feedback": rally_feedback,
            "score_estimate": f"{points_won} - {points_lost}"
        }
