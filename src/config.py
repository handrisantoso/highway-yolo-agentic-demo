"""
Configuration constants for the highway monitoring system.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_NAME = "yolo11n.pt"

# COCO class IDs for vehicles
VEHICLE_CLASSES: Dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
VEHICLE_CLASS_IDS: List[int] = list(VEHICLE_CLASSES.keys())

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
DEFAULT_CONF = 0.35
DEFAULT_IOU = 0.45
DEFAULT_IMGSZ = 640

# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------
MAX_DISAPPEARED = 30       # frames before a track is dropped
MAX_TRACK_HISTORY = 60     # positions kept per track for trajectory analysis
TRACK_DISTANCE_THRESHOLD = 80  # pixels – max centroid distance for association

# ---------------------------------------------------------------------------
# Counting line
# ---------------------------------------------------------------------------
# Normalised position along the major axis [0, 1]
COUNT_LINE_POSITION = 0.5

# ---------------------------------------------------------------------------
# Direction of expected traffic flow
# ---------------------------------------------------------------------------
DIRECTION_LEFT_TO_RIGHT = "left_to_right"
DIRECTION_RIGHT_TO_LEFT = "right_to_left"
DIRECTION_TOP_TO_BOTTOM = "top_to_bottom"
DIRECTION_BOTTOM_TO_TOP = "bottom_to_top"

VALID_DIRECTIONS = [
    DIRECTION_LEFT_TO_RIGHT,
    DIRECTION_RIGHT_TO_LEFT,
    DIRECTION_TOP_TO_BOTTOM,
    DIRECTION_BOTTOM_TO_TOP,
]

# ---------------------------------------------------------------------------
# Abnormal event thresholds
# ---------------------------------------------------------------------------
STOPPED_SPEED_THRESHOLD = 3.0     # px/frame – below this = stopped vehicle
STOPPED_DURATION_FRAMES = 60      # consecutive stopped frames before alert
SPEEDING_THRESHOLD_PX = 25.0     # px/frame – above this = speeding
CONGESTION_DENSITY_THRESHOLD = 8  # vehicles in ROI at once
WRONG_WAY_CONSISTENCY_FRAMES = 10 # frames of wrong-way motion before alert

# ---------------------------------------------------------------------------
# Visualisation colours  (BGR)
# ---------------------------------------------------------------------------
COLOR_VEHICLE = (0, 255, 0)
COLOR_WRONG_WAY = (0, 0, 255)
COLOR_STOPPED = (0, 165, 255)
COLOR_SPEEDING = (255, 0, 255)
COLOR_COUNT_LINE = (255, 255, 0)
COLOR_CONGESTION = (0, 0, 200)
COLOR_TEXT = (255, 255, 255)
COLOR_TEXT_BG = (0, 0, 0)

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
OUTPUT_VIDEO_FPS = 25
OUTPUT_VIDEO_CODEC = "mp4v"
OUTPUT_CSV_NAME = "events.csv"
OUTPUT_COUNTS_NAME = "counts.csv"
