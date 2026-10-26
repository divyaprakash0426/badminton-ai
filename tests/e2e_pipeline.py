"""Headless end-to-end check: YOLO + TrackNet + analysis (+ SlowFast) on a clip.

Usage: python tests/e2e_pipeline.py CLIP.mp4 x1,y1 x2,y2 x3,y3 x4,y4 [out.json]
Corners are TL, TR, BL, BR in pixels.
"""
import json
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.analysis import AnalysisEngine
from engine.geometry import GeometryEngine
from engine.tracker import TrackNetTracker, YOLOTracker
from project_utils.parallel_processor import process_video_concurrent
from project_utils.video import VideoProcessor
from project_utils.video_io import create_h264_writer


class _Progress:
    def progress(self, v):
        print(f"\rprogress {v:5.0%}", end="", flush=True)


def main():
    clip = sys.argv[1]
    corners = [tuple(int(v) for v in a.split(",")) for a in sys.argv[2:6]]
    out_json = sys.argv[6] if len(sys.argv) > 6 else "/tmp/e2e_report.json"

    info = VideoProcessor().get_video_info(clip)
    geometry = GeometryEngine()
    geometry.calculate_homography(corners)

    t0 = time.time()
    yolo, tracknet = YOLOTracker(), TrackNetTracker()
    print(f"models loaded in {time.time() - t0:.1f}s")

    writer = create_h264_writer("/tmp/e2e_out.mp4", info["fps"], (info["width"], info["height"]), queue_size=64)
    t0 = time.time()
    players, shuttle, present, cuts = process_video_concurrent(
        clip, yolo, tracknet, geometry, writer, 0, info["frame_count"], _Progress()
    )
    writer.release()
    print(f"\nprocessed in {time.time() - t0:.1f}s: players={len(players)} shuttle={len(shuttle)} cuts={len(cuts)}")

    report = AnalysisEngine(fps=info["fps"]).analyze_rally(
        players, shuttle, cuts, present, tracknet, geometry, video_path=clip
    )
    with open(out_json, "w") as f:
        json.dump(report, f, default=str, indent=2)
    print("rallies:", len(report.get("rallies", [])), "shots:", report.get("total_shots"))
    print("distribution:", report.get("shot_distribution"))
    assert report.get("rallies"), "no rallies detected"


if __name__ == "__main__":
    main()
