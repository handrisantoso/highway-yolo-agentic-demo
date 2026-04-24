"""
Core pipeline: YOLO11n detection → centroid tracking → event detection →
counting-line logic.

HighwayMonitor is the main orchestration class used by detect_video.py
and the Streamlit app.
"""
from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover
    YOLO = None  # type: ignore

from config import (
    COLOR_CONGESTION,
    COLOR_COUNT_LINE,
    COLOR_SPEEDING,
    COLOR_STOPPED,
    COLOR_TEXT,
    COLOR_TEXT_BG,
    COLOR_VEHICLE,
    COLOR_WRONG_WAY,
    COUNT_LINE_POSITION,
    DEFAULT_CONF,
    DEFAULT_IOU,
    DEFAULT_IMGSZ,
    MODEL_NAME,
    OUTPUT_COUNTS_NAME,
    OUTPUT_CSV_NAME,
    OUTPUT_VIDEO_CODEC,
    OUTPUT_VIDEO_FPS,
    VEHICLE_CLASS_IDS,
    VEHICLE_CLASSES,
)
from event_detection import Event, EventDetector
from tracking_utils import (
    CentroidTracker,
    build_counting_line,
    check_line_crossing,
)


# ---------------------------------------------------------------------------
# HighwayMonitor
# ---------------------------------------------------------------------------

class HighwayMonitor:
    """
    High-level pipeline wrapper.

    Parameters
    ----------
    model_path : str
        Path to the YOLO11n .pt file (downloaded automatically if needed).
    direction : str
        Expected traffic direction, e.g. "left_to_right".
    conf : float
        YOLO confidence threshold.
    iou : float
        YOLO NMS IoU threshold.
    imgsz : int
        Inference image size.
    output_dir : Optional[str]
        If given, CSV and optional video outputs are written here.
    save_video : bool
        Whether to write annotated frames to an output video.
    """

    def __init__(
        self,
        model_path: str = MODEL_NAME,
        direction: str = "left_to_right",
        conf: float = DEFAULT_CONF,
        iou: float = DEFAULT_IOU,
        imgsz: int = DEFAULT_IMGSZ,
        output_dir: Optional[str] = None,
        save_video: bool = False,
    ):
        self.direction = direction
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.output_dir = Path(output_dir) if output_dir else None
        self.save_video = save_video

        # Lazily initialised on first frame (need frame size for counting line)
        self._model: Optional[YOLO] = None
        self._tracker = CentroidTracker()
        self._event_detector = EventDetector(direction)

        self._model_path = model_path

        # Frame & counting state
        self.frame_number = 0
        self.counts: Dict[str, int] = {"in": 0, "out": 0}
        self.all_events: List[Event] = []

        # Counting line (set on first frame)
        self._line_pt1: Optional[Tuple[int, int]] = None
        self._line_pt2: Optional[Tuple[int, int]] = None

        # Video writer
        self._writer: Optional[cv2.VideoWriter] = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, List[Event]]:
        """
        Run the full pipeline on a single BGR frame.

        Returns
        -------
        annotated_frame : np.ndarray
            A copy of the frame with bounding boxes and overlays.
        new_events : List[Event]
            Events fired this frame.
        """
        h, w = frame.shape[:2]

        # One-time init
        if self._model is None:
            self._init_model()
        if self._line_pt1 is None:
            self._line_pt1, self._line_pt2 = build_counting_line(
                w, h, self.direction, COUNT_LINE_POSITION
            )

        # Detect
        detections = self._detect(frame)

        # Track
        tracks = self._tracker.update(detections)

        # Count line crossings
        for tid, track in tracks.items():
            if track.counted:
                continue
            crossing = check_line_crossing(
                track, self._line_pt1, self._line_pt2, self.direction
            )
            if crossing is not None:
                track.counted = True
                track.crossed_direction = crossing
                if crossing == self.direction:
                    self.counts["in"] += 1
                else:
                    self.counts["out"] += 1

        # Events
        new_events = self._event_detector.analyse(tracks, self.frame_number)
        self.all_events.extend(new_events)

        # Annotate
        annotated = self._annotate(frame.copy(), tracks, new_events)

        self.frame_number += 1

        # Write to video if requested
        if self.save_video and self._writer is not None:
            self._writer.write(annotated)

        return annotated, new_events

    def open_video_writer(self, path: str, fps: float, width: int, height: int) -> None:
        fourcc = cv2.VideoWriter_fourcc(*OUTPUT_VIDEO_CODEC)
        self._writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

    def close(self) -> None:
        """Flush CSV outputs and release video writer."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self.output_dir:
            self._save_csv()

    def reset(self) -> None:
        self._tracker.reset()
        self._event_detector.reset()
        self.frame_number = 0
        self.counts = {"in": 0, "out": 0}
        self.all_events.clear()
        self._line_pt1 = None
        self._line_pt2 = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_model(self) -> None:
        if YOLO is None:
            raise RuntimeError(
                "ultralytics is not installed. Run: pip install ultralytics"
            )
        self._model = YOLO(self._model_path)

    def _detect(
        self, frame: np.ndarray
    ) -> List[Tuple[float, float, float, float, int]]:
        """Run YOLO and return vehicle detections as (x1,y1,x2,y2,class_id)."""
        results = self._model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=VEHICLE_CLASS_IDS,
            verbose=False,
        )
        detections = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls = int(box.cls[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append((x1, y1, x2, y2, cls))
        return detections

    def _annotate(
        self,
        frame: np.ndarray,
        tracks: dict,
        new_events: List[Event],
    ) -> np.ndarray:
        h, w = frame.shape[:2]

        # Counting line
        if self._line_pt1 and self._line_pt2:
            cv2.line(frame, self._line_pt1, self._line_pt2, COLOR_COUNT_LINE, 2)

        # Active event track IDs
        event_track_ids = {e.track_id: e.event_type for e in new_events if e.track_id >= 0}

        # Draw tracks
        for tid, track in tracks.items():
            cx, cy = int(track.centroid[0]), int(track.centroid[1])
            event_type = event_track_ids.get(tid)

            color = COLOR_VEHICLE
            if event_type == "WRONG_WAY":
                color = COLOR_WRONG_WAY
            elif event_type == "STOPPED":
                color = COLOR_STOPPED
            elif event_type == "SPEEDING":
                color = COLOR_SPEEDING

            cv2.circle(frame, (cx, cy), 5, color, -1)

            label = f"ID:{tid} {VEHICLE_CLASSES.get(track.class_id, 'veh')}"
            if event_type:
                label += f" [{event_type}]"
            _put_label(frame, label, cx - 5, cy - 10, color)

            # Draw trajectory
            pts = list(track.centroids)
            for i in range(1, len(pts)):
                p1 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
                p2 = (int(pts[i][0]), int(pts[i][1]))
                cv2.line(frame, p1, p2, color, 1)

        # Congestion overlay
        congestion_events = [e for e in new_events if e.event_type == "CONGESTION"]
        if congestion_events:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), COLOR_CONGESTION, -1)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
            _put_label(frame, "! CONGESTION !", w // 2 - 80, 60, COLOR_CONGESTION)

        # Stats panel
        _draw_stats(
            frame,
            frame_n=self.frame_number,
            n_vehicles=len(tracks),
            count_in=self.counts["in"],
            count_out=self.counts["out"],
            direction=self.direction,
        )

        return frame

    def _save_csv(self) -> None:
        assert self.output_dir is not None
        self.output_dir.mkdir(parents=True, exist_ok=True)

        events_path = self.output_dir / OUTPUT_CSV_NAME
        if self.all_events:
            with open(events_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=list(self.all_events[0].to_dict().keys())
                )
                writer.writeheader()
                for ev in self.all_events:
                    writer.writerow(ev.to_dict())

        counts_path = self.output_dir / OUTPUT_COUNTS_NAME
        with open(counts_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["direction", "count"])
            writer.writerow(["in", self.counts["in"]])
            writer.writerow(["out", self.counts["out"]])


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _put_label(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: Tuple[int, int, int],
    font_scale: float = 0.5,
    thickness: int = 1,
) -> None:
    cv2.putText(
        frame, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_TEXT_BG, thickness + 2
    )
    cv2.putText(
        frame, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness
    )


def _draw_stats(
    frame: np.ndarray,
    frame_n: int,
    n_vehicles: int,
    count_in: int,
    count_out: int,
    direction: str,
) -> None:
    lines = [
        f"Frame: {frame_n}",
        f"Vehicles: {n_vehicles}",
        f"Counted IN:  {count_in}",
        f"Counted OUT: {count_out}",
        f"Direction: {direction}",
    ]
    x, y0, dy = 10, 20, 20
    for i, line in enumerate(lines):
        y = y0 + i * dy
        _put_label(frame, line, x, y, COLOR_TEXT)
