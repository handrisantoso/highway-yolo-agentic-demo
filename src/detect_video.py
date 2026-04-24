"""
CLI entry-point for highway video analysis.

Usage
-----
python src/detect_video.py \\
    --video samples/highway.mp4 \\
    --output outputs \\
    --conf 0.35 \\
    --direction left_to_right \\
    --save-video
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running from the repo root or from within src/
sys.path.insert(0, str(Path(__file__).parent))

import cv2

from config import (
    DEFAULT_CONF,
    DEFAULT_IOU,
    DEFAULT_IMGSZ,
    MODEL_NAME,
    VALID_DIRECTIONS,
)
from highway_monitor import HighwayMonitor


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="detect_video.py",
        description="YOLO11n highway monitoring – detect, track, count and report events.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video", required=True,
        help="Path to input video file (or 0 for webcam).",
    )
    parser.add_argument(
        "--output", default="outputs",
        help="Directory for CSV reports and (optionally) the annotated video.",
    )
    parser.add_argument(
        "--model", default=MODEL_NAME,
        help="Path or name of the YOLO model weights.",
    )
    parser.add_argument(
        "--conf", type=float, default=DEFAULT_CONF,
        help="Detection confidence threshold [0, 1].",
    )
    parser.add_argument(
        "--iou", type=float, default=DEFAULT_IOU,
        help="NMS IoU threshold [0, 1].",
    )
    parser.add_argument(
        "--imgsz", type=int, default=DEFAULT_IMGSZ,
        help="Inference image size (pixels).",
    )
    parser.add_argument(
        "--direction", default="left_to_right", choices=VALID_DIRECTIONS,
        help="Expected direction of traffic flow.",
    )
    parser.add_argument(
        "--save-video", action="store_true",
        help="Write annotated frames to an output .mp4 file.",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display annotated frames in a window (requires a display).",
    )
    parser.add_argument(
        "--max-frames", type=int, default=0,
        help="Stop after this many frames (0 = process entire video).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    args = parse_args(argv)

    video_path = args.video
    # Support webcam index
    cap_source: str | int = int(video_path) if video_path.isdigit() else video_path

    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}", file=sys.stderr)
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Video: {video_path}")
    print(f"[INFO] Resolution: {width}x{height}  FPS: {fps:.1f}  Frames: {total_frames}")
    print(f"[INFO] Direction: {args.direction}  Conf: {args.conf}  IoU: {args.iou}")

    monitor = HighwayMonitor(
        model_path=args.model,
        direction=args.direction,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        output_dir=args.output,
        save_video=args.save_video,
    )

    if args.save_video:
        out_video_path = str(Path(args.output) / "highway_output.mp4")
        Path(args.output).mkdir(parents=True, exist_ok=True)
        monitor.open_video_writer(out_video_path, fps, width, height)
        print(f"[INFO] Saving annotated video to: {out_video_path}")

    t0 = time.time()
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated, events = monitor.process_frame(frame)
            frame_count += 1

            # Print any new events to stdout
            for ev in events:
                print(f"  [EVENT] frame={ev.frame_number:06d} {ev.event_type:12s} "
                      f"track={ev.track_id:4d} cls={ev.class_name} {ev.extra}")

            if args.show:
                cv2.imshow("Highway Monitor", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] Quit requested by user.")
                    break

            if args.max_frames and frame_count >= args.max_frames:
                print(f"[INFO] Reached max-frames limit ({args.max_frames}).")
                break

            # Progress
            if frame_count % 100 == 0:
                elapsed = time.time() - t0
                pct = (frame_count / total_frames * 100) if total_frames else 0
                fps_proc = frame_count / elapsed
                print(
                    f"[PROGRESS] {frame_count}/{total_frames} frames "
                    f"({pct:.1f}%)  {fps_proc:.1f} fps"
                )

    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()
        monitor.close()

    elapsed = time.time() - t0
    print(f"\n[DONE] Processed {frame_count} frames in {elapsed:.1f}s "
          f"({frame_count / elapsed:.1f} fps)")
    print(f"[DONE] Counted IN: {monitor.counts['in']}  OUT: {monitor.counts['out']}")
    print(f"[DONE] Total events: {len(monitor.all_events)}")
    if args.output:
        print(f"[DONE] Reports saved to: {args.output}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
