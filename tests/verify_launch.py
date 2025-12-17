import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine.analysis import AnalysisEngine

def test_launch_detection():
    print("Testing Launch Detection...", flush=True)
    
    engine = AnalysisEngine(fps=25.0)
    
    # Create a rally with noise at the start
    # 5 frames of "holding" (noise < 10px movement)
    # Then sudden launch (serve)
    
    noise_frames = []
    start_pos = (100, 100)
    for i in range(5):
        # Jitter around 100,100
        jitter = (np.random.randint(-2, 3), np.random.randint(-2, 3))
        pos = (start_pos[0] + jitter[0], start_pos[1] + jitter[1])
        noise_frames.append({'frame': i, 'pos': pos})
    
    # Launch frames: Serve (Right) then Return (Left) to create a hit event
    launch_frames = []
    # Serve: 5 to 30
    curr_x = 100
    for i in range(5, 31):
        curr_x += 20
        pos = (curr_x, 100)
        launch_frames.append({'frame': i, 'pos': pos})
        
    # Return: 31 to 55 (Sharp turn)
    for i in range(31, 56): # 25 frames
        curr_x -= 20
        pos = (curr_x, 100)
        launch_frames.append({'frame': i, 'pos': pos})
        
    full_rally = noise_frames + launch_frames
    
    # Mock player presence (all frames have "players")
    frames_with_players = set([p['frame'] for p in full_rally])
    
    res = engine.analyze_rally(
        player_history=[], 
        shuttle_history=full_rally,
        player_present_frames=frames_with_players
    )
    
    rallies = res['rallies']
    print(f"Detected {len(rallies)} rallies")
    
    if len(rallies) == 0:
        print("❌ No rallies detected! Launch detection might have filtered everything.")
        return
        
    rally = rallies[0]
    start_frame = rally['start_frame']
    
    # We expect the rally to start around frame 5 (the launch), not frame 0.
    # The 'start_frame' in result is the first 'hit event' frame. 
    # With our logic, we force the *first frame of the trimmed df* to be the hit event.
    
    print(f"Rally Start Frame: {start_frame}")
    
    if start_frame >= 4:
        print("✅ Launch Detection Passed: Trimmed noise frames.")
    else:
        print(f"❌ Failed: Rally started at frame {start_frame}, expected >= 4")
        
if __name__ == "__main__":
    test_launch_detection()
