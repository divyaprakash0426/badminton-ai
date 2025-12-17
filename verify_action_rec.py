import sys
import os

# Mock streamlit for app import if needed (but we probably won't import app, just engine)
# sys.modules['streamlit'] = type('MockStreamlit', (), {'cache_resource': lambda x: x, 'set_page_config': lambda **k: None})

try:
    print("Importing engine.action_recognition...")
    from engine.action_recognition import ShotClassifier
    print("Success.")
    
    print("Instantiating ShotClassifier (CPU)...")
    # This might download the model (SlowFast R50)
    # We'll set a timeout or expect it to be fast enough if internet is available?
    # Actually, slowfast_r50(pretrained=True) downloads from Torch Hub.
    # To avoid long wait/errors if no internet, we can try-catch.
    try:
        classifier = ShotClassifier(device='cpu')
        print("ShotClassifier instantiated.")
    except Exception as e:
        print(f"ShotClassifier instantiation warning (might be network/weights): {e}")

    print("Importing engine.analysis...")
    from engine.analysis import AnalysisEngine
    print("Success.")

    print("Checking AnalysisEngine init...")
    engine = AnalysisEngine()
    if engine.shot_classifier is not None:
        print("AnalysisEngine loaded ShotClassifier successfully.")
    else:
        print("AnalysisEngine passed exception during ShotClassifier load (expected if dependencies missing or mocked).")

except ImportError as e:
    print(f"IMPORT ERROR: {e}")
    sys.exit(1)
except Exception as e:
    print(f"RUNTIME ERROR: {e}")
    sys.exit(1)
