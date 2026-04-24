"""
Abnormal event detection for highway monitoring.

Analyses per-track state to fire events:
  - WRONG_WAY   : vehicle moving against expected traffic direction
  - STOPPED     : vehicle stationary for too long
  - SPEEDING    : vehicle exceeding speed threshold
  - CONGESTION  : vehicle density exceeds threshold
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config import (
    CONGESTION_DENSITY_THRESHOLD,
    DIRECTION_BOTTOM_TO_TOP,
    DIRECTION_LEFT_TO_RIGHT,
    DIRECTION_RIGHT_TO_LEFT,
    DIRECTION_TOP_TO_BOTTOM,
    SPEEDING_THRESHOLD_PX,
    STOPPED_DURATION_FRAMES,
    STOPPED_SPEED_THRESHOLD,
    VEHICLE_CLASSES,
    WRONG_WAY_CONSISTENCY_FRAMES,
)
from tracking_utils import Track


# ---------------------------------------------------------------------------
# Event data class
# ---------------------------------------------------------------------------

@dataclass
class Event:
    event_type: str          # "WRONG_WAY" | "STOPPED" | "SPEEDING" | "CONGESTION"
    track_id: int
    frame_number: int
    timestamp: float = field(default_factory=time.time)
    class_name: str = "vehicle"
    centroid_x: float = 0.0
    centroid_y: float = 0.0
    extra: str = ""

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "track_id": self.track_id,
            "frame_number": self.frame_number,
            "timestamp": self.timestamp,
            "class_name": self.class_name,
            "centroid_x": round(self.centroid_x, 1),
            "centroid_y": round(self.centroid_y, 1),
            "extra": self.extra,
        }


# ---------------------------------------------------------------------------
# Per-track accumulators (not exported)
# ---------------------------------------------------------------------------

class _TrackState:
    def __init__(self):
        self.stopped_frames: int = 0
        self.wrong_way_frames: int = 0
        self.last_wrong_way_event: int = -1000
        self.last_stopped_event: int = -1000
        self.last_speeding_event: int = -1000


# ---------------------------------------------------------------------------
# EventDetector
# ---------------------------------------------------------------------------

class EventDetector:
    """
    Stateful detector that analyses tracks each frame and emits Event objects.
    """

    def __init__(self, expected_direction: str):
        if expected_direction not in (
            DIRECTION_LEFT_TO_RIGHT,
            DIRECTION_RIGHT_TO_LEFT,
            DIRECTION_TOP_TO_BOTTOM,
            DIRECTION_BOTTOM_TO_TOP,
        ):
            raise ValueError(f"Unknown direction: {expected_direction}")

        self.expected_direction = expected_direction
        self._states: Dict[int, _TrackState] = {}

        # Cooldown – don't re-fire the same event too often (in frames)
        self._event_cooldown = 90

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse(
        self,
        tracks: Dict[int, Track],
        frame_number: int,
    ) -> List[Event]:
        """Return a (possibly empty) list of new events for this frame."""
        events: List[Event] = []

        # Ensure state entries exist
        for tid in tracks:
            if tid not in self._states:
                self._states[tid] = _TrackState()

        # Per-track events
        for tid, track in tracks.items():
            state = self._states[tid]
            cx, cy = track.centroid
            cls_name = VEHICLE_CLASSES.get(track.class_id, "vehicle")

            # ---- Wrong-way detection ----
            ww_event = self._check_wrong_way(track, state, frame_number, cls_name)
            if ww_event:
                events.append(ww_event)

            # ---- Stopped vehicle ----
            stopped_event = self._check_stopped(track, state, frame_number, cls_name)
            if stopped_event:
                events.append(stopped_event)

            # ---- Speeding ----
            speed_event = self._check_speeding(track, state, frame_number, cls_name)
            if speed_event:
                events.append(speed_event)

        # ---- Congestion (scene-level) ----
        cong_event = self._check_congestion(tracks, frame_number)
        if cong_event:
            events.append(cong_event)

        # Cleanup dead tracks
        live = set(tracks.keys())
        dead = set(self._states.keys()) - live
        for tid in dead:
            del self._states[tid]

        return events

    def reset(self) -> None:
        self._states.clear()

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_wrong_way(
        self,
        track: Track,
        state: _TrackState,
        frame_number: int,
        cls_name: str,
    ) -> Optional[Event]:
        if len(track.centroids) < 2:
            return None

        vx, vy = track.velocity

        wrong = False
        if self.expected_direction == DIRECTION_LEFT_TO_RIGHT and vx < -1.0:
            wrong = True
        elif self.expected_direction == DIRECTION_RIGHT_TO_LEFT and vx > 1.0:
            wrong = True
        elif self.expected_direction == DIRECTION_TOP_TO_BOTTOM and vy < -1.0:
            wrong = True
        elif self.expected_direction == DIRECTION_BOTTOM_TO_TOP and vy > 1.0:
            wrong = True

        if wrong:
            state.wrong_way_frames += 1
        else:
            state.wrong_way_frames = 0

        if (
            state.wrong_way_frames >= WRONG_WAY_CONSISTENCY_FRAMES
            and (frame_number - state.last_wrong_way_event) >= self._event_cooldown
        ):
            state.last_wrong_way_event = frame_number
            cx, cy = track.centroid
            return Event(
                event_type="WRONG_WAY",
                track_id=track.track_id,
                frame_number=frame_number,
                class_name=cls_name,
                centroid_x=cx,
                centroid_y=cy,
                extra=f"expected={self.expected_direction}",
            )
        return None

    def _check_stopped(
        self,
        track: Track,
        state: _TrackState,
        frame_number: int,
        cls_name: str,
    ) -> Optional[Event]:
        if track.speed < STOPPED_SPEED_THRESHOLD:
            state.stopped_frames += 1
        else:
            state.stopped_frames = 0

        if (
            state.stopped_frames >= STOPPED_DURATION_FRAMES
            and (frame_number - state.last_stopped_event) >= self._event_cooldown
        ):
            state.last_stopped_event = frame_number
            cx, cy = track.centroid
            return Event(
                event_type="STOPPED",
                track_id=track.track_id,
                frame_number=frame_number,
                class_name=cls_name,
                centroid_x=cx,
                centroid_y=cy,
                extra=f"stopped_frames={state.stopped_frames}",
            )
        return None

    def _check_speeding(
        self,
        track: Track,
        state: _TrackState,
        frame_number: int,
        cls_name: str,
    ) -> Optional[Event]:
        if track.speed > SPEEDING_THRESHOLD_PX:
            if (frame_number - state.last_speeding_event) >= self._event_cooldown:
                state.last_speeding_event = frame_number
                cx, cy = track.centroid
                return Event(
                    event_type="SPEEDING",
                    track_id=track.track_id,
                    frame_number=frame_number,
                    class_name=cls_name,
                    centroid_x=cx,
                    centroid_y=cy,
                    extra=f"speed={track.speed:.1f}px/frame",
                )
        return None

    def _check_congestion(
        self,
        tracks: Dict[int, Track],
        frame_number: int,
    ) -> Optional[Event]:
        n = len(tracks)
        if n >= CONGESTION_DENSITY_THRESHOLD:
            return Event(
                event_type="CONGESTION",
                track_id=-1,
                frame_number=frame_number,
                class_name="scene",
                extra=f"vehicles={n}",
            )
        return None
