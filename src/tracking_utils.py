"""
Centroid-based multi-object tracker for highway monitoring.

Assigns stable IDs to detected bounding boxes across frames using a simple
nearest-centroid assignment strategy (no external dependency required).
"""
from __future__ import annotations

import math
from collections import OrderedDict, deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import MAX_DISAPPEARED, MAX_TRACK_HISTORY, TRACK_DISTANCE_THRESHOLD


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class Track:
    """State for a single tracked object."""

    def __init__(self, track_id: int, centroid: Tuple[float, float], class_id: int):
        self.track_id = track_id
        self.class_id = class_id
        self.centroids: deque[Tuple[float, float]] = deque(maxlen=MAX_TRACK_HISTORY)
        self.centroids.append(centroid)
        self.disappeared = 0
        self.counted = False          # whether it crossed the counting line
        self.crossed_direction: Optional[str] = None

    @property
    def centroid(self) -> Tuple[float, float]:
        return self.centroids[-1]

    @property
    def velocity(self) -> Tuple[float, float]:
        """Velocity as (dx, dy) over the last two positions."""
        if len(self.centroids) < 2:
            return (0.0, 0.0)
        prev = self.centroids[-2]
        curr = self.centroids[-1]
        return (curr[0] - prev[0], curr[1] - prev[1])

    @property
    def speed(self) -> float:
        vx, vy = self.velocity
        return math.hypot(vx, vy)

    def update(self, centroid: Tuple[float, float], class_id: int) -> None:
        self.centroids.append(centroid)
        self.class_id = class_id
        self.disappeared = 0


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class CentroidTracker:
    """Assign persistent IDs to detections via centroid proximity."""

    def __init__(
        self,
        max_disappeared: int = MAX_DISAPPEARED,
        max_distance: float = TRACK_DISTANCE_THRESHOLD,
    ):
        self.next_id = 0
        self.tracks: OrderedDict[int, Track] = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        detections: List[Tuple[float, float, float, float, int]],
    ) -> Dict[int, Track]:
        """
        Update tracker with a list of (x1, y1, x2, y2, class_id) detections.
        Returns the current live tracks dict {track_id: Track}.
        """
        if not detections:
            self._age_tracks()
            return self.tracks

        centroids = [
            ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            for x1, y1, x2, y2, _ in detections
        ]

        if not self.tracks:
            for (cx, cy), (_, _, _, _, cls) in zip(centroids, detections):
                self._register(cx, cy, cls)
            return self.tracks

        # Build cost matrix
        track_ids = list(self.tracks.keys())
        track_centroids = [self.tracks[tid].centroid for tid in track_ids]

        cost = np.zeros((len(track_centroids), len(centroids)), dtype=float)
        for r, tc in enumerate(track_centroids):
            for c, dc in enumerate(centroids):
                cost[r, c] = math.hypot(tc[0] - dc[0], tc[1] - dc[1])

        # Greedy assignment (row = track, col = detection)
        rows = cost.min(axis=1).argsort()
        cols = cost.argmin(axis=1)

        used_rows: set = set()
        used_cols: set = set()

        for row in rows:
            col = cols[row]
            if row in used_rows or col in used_cols:
                continue
            if cost[row, col] > self.max_distance:
                continue
            tid = track_ids[row]
            cx, cy = centroids[col]
            cls = detections[col][4]
            self.tracks[tid].update((cx, cy), cls)
            used_rows.add(row)
            used_cols.add(col)

        # Unmatched tracks → age / remove
        for row in set(range(len(track_ids))) - used_rows:
            tid = track_ids[row]
            self.tracks[tid].disappeared += 1
            if self.tracks[tid].disappeared > self.max_disappeared:
                del self.tracks[tid]

        # Unmatched detections → register
        for col in set(range(len(detections))) - used_cols:
            cx, cy = centroids[col]
            cls = detections[col][4]
            self._register(cx, cy, cls)

        return self.tracks

    def reset(self) -> None:
        self.next_id = 0
        self.tracks.clear()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _register(self, cx: float, cy: float, class_id: int) -> None:
        self.tracks[self.next_id] = Track(self.next_id, (cx, cy), class_id)
        self.next_id += 1

    def _age_tracks(self) -> None:
        to_delete = []
        for tid, track in self.tracks.items():
            track.disappeared += 1
            if track.disappeared > self.max_disappeared:
                to_delete.append(tid)
        for tid in to_delete:
            del self.tracks[tid]


# ---------------------------------------------------------------------------
# Counting-line helper
# ---------------------------------------------------------------------------

def build_counting_line(
    frame_width: int,
    frame_height: int,
    direction: str,
    position: float = 0.5,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Return (pt1, pt2) for a counting line perpendicular to traffic flow.

    For horizontal traffic (left/right): a vertical line.
    For vertical traffic (top/bottom):   a horizontal line.
    """
    from config import (
        DIRECTION_LEFT_TO_RIGHT,
        DIRECTION_RIGHT_TO_LEFT,
        DIRECTION_TOP_TO_BOTTOM,
        DIRECTION_BOTTOM_TO_TOP,
    )

    if direction in (DIRECTION_LEFT_TO_RIGHT, DIRECTION_RIGHT_TO_LEFT):
        x = int(frame_width * position)
        return (x, 0), (x, frame_height)
    else:
        y = int(frame_height * position)
        return (0, y), (frame_width, y)


def check_line_crossing(
    track: Track,
    line_pt1: Tuple[int, int],
    line_pt2: Tuple[int, int],
    direction: str,
) -> Optional[str]:
    """
    Return the crossing direction string if the track just crossed the line,
    otherwise None.  Uses the sign-change of the signed distance to the line.
    """
    from config import (
        DIRECTION_LEFT_TO_RIGHT,
        DIRECTION_RIGHT_TO_LEFT,
        DIRECTION_TOP_TO_BOTTOM,
        DIRECTION_BOTTOM_TO_TOP,
    )

    if len(track.centroids) < 2:
        return None

    cx_prev, cy_prev = track.centroids[-2]
    cx_curr, cy_curr = track.centroids[-1]

    lx1, ly1 = line_pt1
    lx2, ly2 = line_pt2

    def signed_dist(px, py):
        # Cross product of line vector and point vector
        return (lx2 - lx1) * (py - ly1) - (ly2 - ly1) * (px - lx1)

    d_prev = signed_dist(cx_prev, cy_prev)
    d_curr = signed_dist(cx_curr, cy_curr)

    if d_prev * d_curr >= 0:
        return None  # no crossing

    # Determine crossing direction
    if direction in (DIRECTION_LEFT_TO_RIGHT, DIRECTION_RIGHT_TO_LEFT):
        if cx_curr > cx_prev:
            return DIRECTION_LEFT_TO_RIGHT
        else:
            return DIRECTION_RIGHT_TO_LEFT
    else:
        if cy_curr > cy_prev:
            return DIRECTION_TOP_TO_BOTTOM
        else:
            return DIRECTION_BOTTOM_TO_TOP
