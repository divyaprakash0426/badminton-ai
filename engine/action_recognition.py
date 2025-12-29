import torch
# --- Compatibility Patch for torchvision > 0.18 ---
import sys
import torchvision
try:
    import torchvision.transforms.functional_tensor
except ImportError:
    # pytorchvideo relies on this deprecated module
    import torchvision.transforms.functional as F
    sys.modules["torchvision.transforms.functional_tensor"] = F
# --------------------------------------------------
import torch.nn as nn
from torchvision.transforms import Compose, Lambda, CenterCrop
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    Normalize,
)
from pytorchvideo.models.hub import slowfast_r50
import numpy as np

class ShotClassifier:
    def __init__(self, model_path='/home/modernyogi/Projects/badminton-ai/ckpts/best_shot_classifier.pth', num_classes=18, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the ShotClassifier with a SlowFast model.
        
        Args:
            model_path (str, optional): Path to local weights. If None, loads pretrained Kinetics-400 weights.
            num_classes (int): Number of classes (e.g., 18 for VideoBadminton).
            device (str): 'cuda' or 'cpu'.
        """
        self.device = device
        self.num_classes = num_classes
        
        # Load SlowFast R50 model
        # We start with pretrained weights on Kinetics-400 for better feature extraction
        self.model = slowfast_r50(pretrained=True)
        
        # Modify the head for our specific number of classes
        # The original head has 400 classes (Kinetics-400)
        self.model.blocks[6].proj = nn.Linear(self.model.blocks[6].proj.in_features, num_classes)
        
        if model_path:
            try:
                # Security: Use weights_only=True to prevent arbitrary code execution during deserialization
                state_dict = torch.load(model_path, map_location=device, weights_only=True)
                self.model.load_state_dict(state_dict)
                print(f"Loaded ShotClassifier weights from {model_path}")
            except Exception as e:
                print(f"Error loading weights from {model_path}: {e}")
                print("Using initialized weights (ensure you train the model first!)")
        
        self.model.to(self.device)
        self.model.eval()

        # Define transforms
        # SlowFast expects a list of two tensors: [slow_path, fast_path]
        # But for inference on raw frames, we first need to transform the input tensor
        self.transforms = Compose([
            Lambda(lambda x: x / 255.0), # Normalize [0, 255] -> [0, 1]
            Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)), # ImageNet mean/std approximation
            ShortSideScale(size=256),
            CenterCrop(256)
        ])
        
        # Parameters for SlowFast input
        self.alpha = 4
        self.num_frames = 32 # Input frame count
        self.sampling_rate = 2
        
        # Class mapping for VideoBadminton (Sorted by folder name)
        self.classes = [
            "00_Short Serve",
            "01_Cross Court Flight",
            "02_Lift",
            "03_Tap Smash",
            "04_Block",
            "05_Drop Shot",
            "06_Push Shot",
            "07_Transitional Slice",
            "08_Cut",
            "09_Rush Shot",
            "10_Defensive Clear",
            "11_Defensive Drive",
            "12_Clear",
            "13_Long Serve",
            "14_Smash",
            "15_Flat Shot",
            "16_Rear Court Flat Drive",
            "17_Short Flat Shot"
        ] 
        # Note: The user should update this list to match exactly what their training dataset produces.

    def _pack_pathway_output(self, frames):
        """
        Prepare input for SlowFast (Slow and Fast pathways).
        frames: shape (C, T, H, W)
        """
        fast_pathway = frames
        
        # Determine time dimension based on input shape
        # (B, C, T, H, W) -> Dim 2 is Time
        # (C, T, H, W)    -> Dim 1 is Time
        if frames.ndim == 5:
            time_dim = 2
        elif frames.ndim == 4:
            time_dim = 1
        else:
            raise ValueError(f"Expected frames to be 4D or 5D, got {frames.shape}")
            
        # Sample frames for the slow pathway
        indices = torch.linspace(
            0, frames.shape[time_dim] - 1, frames.shape[time_dim] // self.alpha
        ).long().to(frames.device)
        
        slow_pathway = torch.index_select(frames, time_dim, indices)
        
        return [slow_pathway, fast_pathway]

    def predict(self, video_tensor):
        """
        Run inference on a video clip.
        
        Args:
            video_tensor (torch.Tensor or np.ndarray): 
                Shape (T, H, W, C) for numpy (0-255)
                or (C, T, H, W) for torch.
                Ideally T should be around 32-64 frames.
        
        Returns:
            str: Predicted class name.
            float: Confidence score.
        """
        if isinstance(video_tensor, np.ndarray):
            # Convert (T, H, W, C) -> (C, T, H, W)
            video_tensor = torch.from_numpy(video_tensor).permute(3, 0, 1, 2).float()
            
        # Ensure correct temporal length
        # For simplicity, we assume the input is roughly correct or we subsample/pad
        # Ideally, use UniformTemporalSubsample(self.num_frames) if T is large
        if video_tensor.shape[1] != self.num_frames:
             transform = UniformTemporalSubsample(self.num_frames)
             video_tensor = transform(video_tensor)

        video_tensor = video_tensor.to(self.device)
        
        # Apply transforms
        video_tensor = self.transforms(video_tensor)
        
        # Add batch dimension (C, T, H, W) -> (1, C, T, H, W)
        video_tensor = video_tensor.unsqueeze(0)
        
        # Prepare inputs
        inputs = self._pack_pathway_output(video_tensor)
        
        with torch.no_grad():
            preds = self.model(inputs)
            
        # Get top prediction
        post_act = torch.nn.Softmax(dim=1)
        preds = post_act(preds)
        conf, pred_idx_tensor = preds.max(1)
        pred_idx = pred_idx_tensor.item()
        
        if pred_idx < len(self.classes):
            return self.classes[pred_idx], conf.item()
        else:
            return "Unknown", 0.0
