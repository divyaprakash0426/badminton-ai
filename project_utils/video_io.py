"""
Async Video I/O Utilities for GPU-accelerated video processing.

Features:
- AsyncVideoWriter: Background thread for non-blocking frame writing
- GPUVideoReader: NVDEC hardware-accelerated video decoding (if available)
"""

import cv2
import numpy as np
import threading
import queue
import subprocess
import os


class AsyncVideoWriter:
    """
    Asynchronous video writer using a background thread.
    Writes frames without blocking the main processing loop.
    """
    
    def __init__(self, output_path, fourcc, fps, frame_size, queue_size=64):
        """
        Args:
            output_path: Path to output video file
            fourcc: FourCC codec code (e.g., cv2.VideoWriter_fourcc(*'mp4v'))
            fps: Frames per second
            frame_size: Tuple of (width, height)
            queue_size: Max frames to buffer (default 64)
        """
        self.output_path = output_path
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.running = True
        
        # Start background writer thread
        self.writer_thread = threading.Thread(target=self._write_loop, daemon=True)
        self.writer_thread.start()
        print(f"AsyncVideoWriter started: {output_path} ({queue_size} frame buffer)")
    
    def _write_loop(self):
        """Background thread that writes frames from queue."""
        while self.running or not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get(timeout=0.1)
                self.writer.write(frame)
                self.frame_queue.task_done()
            except queue.Empty:
                continue
    
    def write(self, frame):
        """
        Queue a frame for writing (non-blocking).
        If queue is full, blocks until space is available.
        """
        self.frame_queue.put(frame)
    
    def release(self):
        """Stop the writer and wait for all frames to be written."""
        self.running = False
        self.frame_queue.join()  # Wait for queue to empty
        self.writer_thread.join(timeout=5)
        self.writer.release()
        print(f"AsyncVideoWriter finished: {self.output_path}")


class GPUVideoReader:
    """
    GPU-accelerated video reader using FFmpeg with NVDEC.
    Falls back to OpenCV if NVDEC is not available.
    """
    
    def __init__(self, video_path, use_gpu=True):
        """
        Args:
            video_path: Path to input video
            use_gpu: If True, attempt to use NVDEC hardware decoding
        """
        self.video_path = video_path
        self.use_gpu = use_gpu and self._check_nvdec()
        self.process = None
        self.cap = None
        
        # Get video info
        self.cap = cv2.VideoCapture(video_path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.cap.release()
        self.cap = None
        
        # NVDEC GPU decoding - fixed! (removed -hwaccel_output_format cuda that broke stdout)
        if self.use_gpu:
            self._init_nvdec()
        else:
            self._init_opencv()
    
    def _check_nvdec(self):
        """Check if NVDEC is available via FFmpeg."""
        try:
            result = subprocess.run(
                ['ffmpeg', '-hwaccels'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return 'cuda' in result.stdout.lower()
        except:
            return False
    
    def _init_nvdec(self):
        """Initialize FFmpeg with NVDEC hardware acceleration.
        
        Note: We use -hwaccel cuda WITHOUT -hwaccel_output_format cuda.
        This decodes on GPU but transfers frames to CPU for stdout output.
        """
        cmd = [
            'ffmpeg',
            '-hwaccel', 'cuda',  # Use GPU for decoding
            # NOTE: Do NOT use -hwaccel_output_format cuda - it keeps frames in GPU memory
            # which cannot be piped to stdout
            '-i', self.video_path,
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-v', 'quiet',
            '-'
        ]
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=self.width * self.height * 3 * 4  # Buffer 4 frames
            )
            print(f"GPUVideoReader: Using NVDEC hardware decoding (GPU→CPU)")
        except Exception as e:
            print(f"NVDEC init failed: {e}. Falling back to OpenCV.")
            self.use_gpu = False
            self._init_opencv()
    
    def _init_opencv(self):
        """Initialize standard OpenCV video capture."""
        self.cap = cv2.VideoCapture(self.video_path)
        print(f"GPUVideoReader: Using OpenCV (CPU decoding)")
    
    def read(self):
        """
        Read the next frame.
        
        Returns:
            Tuple of (success: bool, frame: np.ndarray or None)
        """
        if self.use_gpu and self.process:
            raw = self.process.stdout.read(self.width * self.height * 3)
            if len(raw) != self.width * self.height * 3:
                return False, None
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3))
            return True, frame
        elif self.cap:
            return self.cap.read()
        return False, None
    
    def read_batch(self, batch_size):
        """
        Read a batch of frames efficiently.
        
        Args:
            batch_size: Number of frames to read
            
        Returns:
            List of frames (may be less than batch_size if video ends)
        """
        frames = []
        for _ in range(batch_size):
            ret, frame = self.read()
            if not ret:
                break
            frames.append(frame)
        return frames
    
    def seek(self, frame_idx):
        """
        Seek to a specific frame index.
        Note: Only works efficiently with OpenCV fallback.
        """
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        elif self.use_gpu:
            # For NVDEC, we need to restart with -ss
            self.release()
            start_time = frame_idx / self.fps
            cmd = [
                'ffmpeg',
                '-hwaccel', 'cuda',
                '-hwaccel_output_format', 'cuda',
                '-ss', str(start_time),
                '-i', self.video_path,
                '-f', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-v', 'quiet',
                '-'
            ]
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=self.width * self.height * 3 * 4
            )
    
    def isOpened(self):
        """Check if reader is open and ready."""
        if self.use_gpu:
            return self.process is not None and self.process.poll() is None
        return self.cap is not None and self.cap.isOpened()
    
    def release(self):
        """Release resources."""
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None
        if self.cap:
            self.cap.release()
            self.cap = None


def create_video_reader(video_path, use_gpu=True, start_frame=0):
    """
    Factory function to create the best available video reader.
    
    Args:
        video_path: Path to video file
        use_gpu: Try to use GPU decoding if available
        start_frame: Frame index to start reading from
        
    Returns:
        GPUVideoReader instance
    """
    reader = GPUVideoReader(video_path, use_gpu=use_gpu)
    if start_frame > 0:
        reader.seek(start_frame)
    return reader


def create_async_writer(output_path, fps, frame_size, queue_size=64):
    """
    Factory function to create async video writer.
    
    Args:
        output_path: Path to output video
        fps: Frames per second
        frame_size: Tuple of (width, height)
        queue_size: Frame buffer size
        
    Returns:
        AsyncVideoWriter instance
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return AsyncVideoWriter(output_path, fourcc, fps, frame_size, queue_size)


def create_h264_writer(output_path, fps, frame_size, queue_size=64):
    """
    Create async video writer with H.264 codec (web-compatible, no transcoding needed).
    
    Tries multiple H.264 codecs in order:
    1. avc1 (native H.264, best compatibility)
    2. x264 (libx264, if available)
    3. mp4v (MPEG-4, fallback)
    
    Args:
        output_path: Path to output video (should end in .mp4)
        fps: Frames per second
        frame_size: Tuple of (width, height)
        queue_size: Frame buffer size
        
    Returns:
        AsyncVideoWriter instance with H.264 encoding
    """
    # Try H.264 codecs in order of preference
    codecs = [
        ('avc1', 'H.264/AVC'),  # Native H.264
        ('H264', 'H.264'),       # Alternative
        ('x264', 'libx264'),     # OpenCV with x264
        ('mp4v', 'MPEG-4'),      # Fallback (requires transcoding)
    ]
    
    for codec_fourcc, codec_name in codecs:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec_fourcc)
            # Test if codec works by creating a test writer
            test_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
            if test_writer.isOpened():
                test_writer.release()
                print(f"H264Writer: Using {codec_name} codec ({codec_fourcc})")
                return AsyncVideoWriter(output_path, fourcc, fps, frame_size, queue_size)
            test_writer.release()
        except Exception as e:
            continue
    
    # Ultimate fallback
    print("H264Writer: No H.264 codec found, using mp4v (will need transcoding)")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return AsyncVideoWriter(output_path, fourcc, fps, frame_size, queue_size)

