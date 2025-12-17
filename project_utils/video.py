import cv2
import numpy as np
import tempfile
import os

class VideoProcessor:
    def __init__(self):
        pass

    def get_video_info(self, video_path):
        """
        Returns basic video information.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        cap.release()
        return {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count
        }

    def get_first_frame(self, video_path):
        """
        Returns the first frame of the video.
        """
        return self.get_frame(video_path, 0)

    def get_frame(self, video_path, frame_index):
        """
        Returns the frame at the specified index.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        # Seek
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Convert to RGB for display
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None
