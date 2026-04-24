"""
Basic unit tests for the highway monitoring system.

These tests are designed to run without a GPU and without downloading any model
weights, so they only validate pure-Python logic.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

class TestConfig:
    def test_vehicle_classes_non_empty(self):
        from config import VEHICLE_CLASSES
        assert len(VEHICLE_CLASSES) > 0

    def test_valid_directions_contains_expected(self):
        from config import VALID_DIRECTIONS
        assert "left_to_right" in VALID_DIRECTIONS
        assert "right_to_left" in VALID_DIRECTIONS
        assert "top_to_bottom" in VALID_DIRECTIONS
        assert "bottom_to_top" in VALID_DIRECTIONS

    def test_thresholds_positive(self):
        from config import (
            DEFAULT_CONF,
            DEFAULT_IOU,
            STOPPED_SPEED_THRESHOLD,
            SPEEDING_THRESHOLD_PX,
            CONGESTION_DENSITY_THRESHOLD,
        )
        assert 0 < DEFAULT_CONF < 1
        assert 0 < DEFAULT_IOU < 1
        assert STOPPED_SPEED_THRESHOLD > 0
        assert SPEEDING_THRESHOLD_PX > STOPPED_SPEED_THRESHOLD
        assert CONGESTION_DENSITY_THRESHOLD > 0


# ---------------------------------------------------------------------------
# tracking_utils – Track
# ---------------------------------------------------------------------------

class TestTrack:
    def _make_track(self, centroid=(100.0, 200.0), class_id=2):
        from tracking_utils import Track
        return Track(track_id=0, centroid=centroid, class_id=class_id)

    def test_initial_centroid(self):
        t = self._make_track((50.0, 60.0))
        assert t.centroid == (50.0, 60.0)

    def test_velocity_single_position(self):
        t = self._make_track()
        assert t.velocity == (0.0, 0.0)

    def test_velocity_after_update(self):
        t = self._make_track((0.0, 0.0))
        t.update((3.0, 4.0), class_id=2)
        assert t.velocity == (3.0, 4.0)
        assert math.isclose(t.speed, 5.0)

    def test_update_resets_disappeared(self):
        t = self._make_track()
        t.disappeared = 5
        t.update((10.0, 10.0), class_id=2)
        assert t.disappeared == 0

    def test_track_history_bounded(self):
        from config import MAX_TRACK_HISTORY
        from tracking_utils import Track
        t = Track(0, (0.0, 0.0), 2)
        for i in range(MAX_TRACK_HISTORY + 10):
            t.update((float(i), 0.0), 2)
        assert len(t.centroids) <= MAX_TRACK_HISTORY


# ---------------------------------------------------------------------------
# tracking_utils – CentroidTracker
# ---------------------------------------------------------------------------

class TestCentroidTracker:
    def test_empty_update_returns_empty(self):
        from tracking_utils import CentroidTracker
        tracker = CentroidTracker()
        tracks = tracker.update([])
        assert tracks == {}

    def test_registers_new_detection(self):
        from tracking_utils import CentroidTracker
        tracker = CentroidTracker()
        tracks = tracker.update([(0, 0, 100, 100, 2)])
        assert len(tracks) == 1

    def test_assigns_same_id_on_close_detection(self):
        from tracking_utils import CentroidTracker
        tracker = CentroidTracker()
        tracks1 = tracker.update([(0, 0, 100, 100, 2)])
        tid1 = list(tracks1.keys())[0]
        # Move detection slightly
        tracks2 = tracker.update([(5, 5, 105, 105, 2)])
        tid2 = list(tracks2.keys())[0]
        assert tid1 == tid2

    def test_assigns_new_id_on_far_detection(self):
        from tracking_utils import CentroidTracker
        tracker = CentroidTracker(max_distance=10)
        tracker.update([(0, 0, 10, 10, 2)])
        tracks = tracker.update([(500, 500, 600, 600, 2)])
        assert len(tracks) == 2

    def test_removes_disappeared_track(self):
        from tracking_utils import CentroidTracker
        tracker = CentroidTracker(max_disappeared=2)
        tracker.update([(0, 0, 100, 100, 2)])
        tracker.update([])  # frame 2 – disappeared=1
        tracker.update([])  # frame 3 – disappeared=2
        tracks = tracker.update([])  # frame 4 – disappeared=3 → removed
        assert len(tracks) == 0

    def test_reset_clears_state(self):
        from tracking_utils import CentroidTracker
        tracker = CentroidTracker()
        tracker.update([(0, 0, 100, 100, 2)])
        tracker.reset()
        assert len(tracker.tracks) == 0
        assert tracker.next_id == 0


# ---------------------------------------------------------------------------
# tracking_utils – counting line helpers
# ---------------------------------------------------------------------------

class TestCountingLine:
    def test_horizontal_flow_gives_vertical_line(self):
        from tracking_utils import build_counting_line
        pt1, pt2 = build_counting_line(800, 600, "left_to_right", 0.5)
        # Vertical line: same x, different y
        assert pt1[0] == pt2[0] == 400
        assert pt1[1] == 0
        assert pt2[1] == 600

    def test_vertical_flow_gives_horizontal_line(self):
        from tracking_utils import build_counting_line
        pt1, pt2 = build_counting_line(800, 600, "top_to_bottom", 0.5)
        assert pt1[1] == pt2[1] == 300
        assert pt1[0] == 0
        assert pt2[0] == 800

    def test_line_crossing_detected(self):
        from tracking_utils import Track, build_counting_line, check_line_crossing
        pt1, pt2 = build_counting_line(800, 600, "left_to_right", 0.5)
        t = Track(0, (350.0, 300.0), 2)
        t.update((450.0, 300.0), 2)
        result = check_line_crossing(t, pt1, pt2, "left_to_right")
        assert result == "left_to_right"

    def test_no_crossing_same_side(self):
        from tracking_utils import Track, build_counting_line, check_line_crossing
        pt1, pt2 = build_counting_line(800, 600, "left_to_right", 0.5)
        t = Track(0, (100.0, 300.0), 2)
        t.update((200.0, 300.0), 2)
        result = check_line_crossing(t, pt1, pt2, "left_to_right")
        assert result is None


# ---------------------------------------------------------------------------
# event_detection – EventDetector
# ---------------------------------------------------------------------------

class TestEventDetector:
    def _make_detector(self, direction="left_to_right"):
        from event_detection import EventDetector
        return EventDetector(direction)

    def test_invalid_direction_raises(self):
        from event_detection import EventDetector
        with pytest.raises(ValueError):
            EventDetector("diagonal")

    def test_no_events_on_empty_tracks(self):
        det = self._make_detector()
        events = det.analyse({}, frame_number=0)
        assert events == []

    def test_wrong_way_detected(self):
        from config import WRONG_WAY_CONSISTENCY_FRAMES
        from tracking_utils import Track
        det = self._make_detector("left_to_right")

        # Build a track moving right-to-left (wrong way)
        t = Track(0, (500.0, 300.0), 2)
        for _ in range(WRONG_WAY_CONSISTENCY_FRAMES + 2):
            t.centroids[-1]  # touch
            t.update((t.centroid[0] - 10.0, t.centroid[1]), 2)

        events = det.analyse({0: t}, frame_number=WRONG_WAY_CONSISTENCY_FRAMES + 5)
        types = [e.event_type for e in events]
        assert "WRONG_WAY" in types

    def test_stopped_vehicle_detected(self):
        from config import STOPPED_DURATION_FRAMES
        from tracking_utils import Track
        det = self._make_detector()

        t = Track(0, (300.0, 300.0), 2)
        # Simulate stationary track for enough frames
        for _ in range(STOPPED_DURATION_FRAMES + 2):
            t.update((300.0, 300.0), 2)

        events = det.analyse({0: t}, frame_number=STOPPED_DURATION_FRAMES + 5)
        types = [e.event_type for e in events]
        assert "STOPPED" in types

    def test_speeding_vehicle_detected(self):
        from config import SPEEDING_THRESHOLD_PX
        from tracking_utils import Track
        det = self._make_detector()

        t = Track(0, (0.0, 300.0), 2)
        t.update((SPEEDING_THRESHOLD_PX + 5.0, 300.0), 2)

        events = det.analyse({0: t}, frame_number=10)
        types = [e.event_type for e in events]
        assert "SPEEDING" in types

    def test_congestion_detected(self):
        from config import CONGESTION_DENSITY_THRESHOLD
        from tracking_utils import Track
        det = self._make_detector()

        tracks = {
            i: Track(i, (float(i * 50), 300.0), 2)
            for i in range(CONGESTION_DENSITY_THRESHOLD)
        }
        events = det.analyse(tracks, frame_number=5)
        types = [e.event_type for e in events]
        assert "CONGESTION" in types

    def test_event_to_dict(self):
        import time as _time
        from event_detection import Event
        ev = Event(
            event_type="STOPPED",
            track_id=3,
            frame_number=42,
            class_name="car",
            centroid_x=100.0,
            centroid_y=200.0,
        )
        d = ev.to_dict()
        assert d["event_type"] == "STOPPED"
        assert d["track_id"] == 3
        assert d["frame_number"] == 42
        assert "timestamp" in d


# ---------------------------------------------------------------------------
# detect_video – argument parser
# ---------------------------------------------------------------------------

class TestDetectVideoArgs:
    def test_default_args(self):
        from detect_video import parse_args
        args = parse_args(["--video", "samples/highway.mp4"])
        assert args.video == "samples/highway.mp4"
        assert args.conf == pytest.approx(0.35, abs=1e-6)
        assert args.direction == "left_to_right"
        assert args.save_video is False

    def test_save_video_flag(self):
        from detect_video import parse_args
        args = parse_args([
            "--video", "test.mp4",
            "--output", "out/",
            "--save-video",
        ])
        assert args.save_video is True
        assert args.output == "out/"

    def test_invalid_direction_raises(self):
        from detect_video import parse_args
        with pytest.raises(SystemExit):
            parse_args(["--video", "v.mp4", "--direction", "diagonal"])
