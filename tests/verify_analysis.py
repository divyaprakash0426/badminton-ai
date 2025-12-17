import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine.analysis import AnalysisEngine
from engine.geometry import GeometryEngine

def test_analysis_fixes():
    print("Testing AnalysisEngine Fixes...")
    
    # 1. Test Initialization
    fps = 30.0
    engine = AnalysisEngine(fps=fps)
    assert engine.fps == 30.0, "FPS not set correctly"
    print("✅ Initialization Passed")
    
    # 2. Test Metric-Based Classification (Mocking Geometry)
    print("\nTesting Shot Classification with Metrics...")
    
    # Mock Geometry Engine
    class MockGeometry:
        def __init__(self):
            self.matrix = np.eye(3) # Dummy
        def transform_point(self, p):
            # Simple scaling: 100px = 1m for testing
            return (p[0]/100.0, p[1]/100.0)
            
    geo = MockGeometry()
    
    # Case A: Smash (Fast Speed)
    # Start: (100, 100) -> 1m, 1m (Far court?)
    # End: (300, 300) -> 3m, 3m
    # Dist: sqrt(2^2 + 2^2) = 2.82m
    # Time: 0.05s (very fast) -> Speed = 56 m/s
    start_evt = {'frame': 0, 'pos': (100, 100)}
    end_evt = {'frame': 2, 'pos': (300, 300)} # 2 frames at 30fps = 0.066s
    
    # Mock Dataframe for segment
    df_smash = pd.DataFrame([
        {'frame': 0, 'pos': (100, 100)},
        {'frame': 1, 'pos': (200, 200)},
        {'frame': 2, 'pos': (300, 300)}
    ])
    
    shot_type = engine._classify_single_shot(start_evt, end_evt, df_smash, geometry=geo)
    print(f"Smash Test detected as: {shot_type}")
    assert shot_type == "Smash", f"Expected Smash, got {shot_type}"
    print("✅ Smash Detection Passed")
    
    # Case B: Drop (Slow Speed, Lands near Net)
    # Net at 6.7m (670px in our mock scale).
    # Start: (100, 100) -> 1m, 1m
    # End: (100, 650) -> 1m, 6.5m (Near Net)
    # Dist: 5.5m
    # Time: 1.0s (30 frames) -> Speed = 5.5 m/s
    start_drop = {'frame': 10, 'pos': (100, 100)}
    end_drop = {'frame': 40, 'pos': (100, 650)}
    
    df_drop = pd.DataFrame([
        {'frame': 10, 'pos': (100, 100)},
        {'frame': 25, 'pos': (100, 300)}, # Midpoint
        {'frame': 40, 'pos': (100, 650)}
    ])
    
    shot_type_drop = engine._classify_single_shot(start_drop, end_drop, df_drop, geometry=geo)
    print(f"Drop Test detected as: {shot_type_drop}")
    assert shot_type_drop == "Drop", f"Expected Drop, got {shot_type_drop}"
    print("✅ Drop Detection Passed")
    
    # Case C: Clear (Slow Speed, Lands Far)
    # End: (100, 1200) -> 12m (Far back)
    end_clear = {'frame': 40, 'pos': (100, 1200)}
    df_clear = pd.DataFrame([
        {'frame': 10, 'pos': (100, 100)},
        {'frame': 40, 'pos': (100, 1200)}
    ])
    shot_type_clear = engine._classify_single_shot(start_drop, end_clear, df_clear, geometry=geo) # Same start
    print(f"Clear Test detected as: {shot_type_clear}")
    assert shot_type_clear == "Clear", f"Expected Clear, got {shot_type_clear}"
    print("✅ Clear Detection Passed")

    print("\nAll Tests Passed!")

if __name__ == "__main__":
    test_analysis_fixes()
