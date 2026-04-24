# Highway YOLO11n Monitoring System

A complete Python pipeline for real-time highway surveillance using **YOLO11n** — the
nano variant of Ultralytics YOLO11.

## Features

| Feature | Details |
|---|---|
| Object detection | Cars, motorcycles, buses and trucks via YOLO11n |
| Vehicle counting | Bi-directional crossing-line counter |
| Wrong-way detection | Flags vehicles moving against the expected flow |
| Abnormal event detection | Stopped vehicles, speeding, traffic congestion |
| Streamlit dashboard | Interactive web UI with live feed and charts |
| CSV reports | Per-event and per-direction count exports |

---

## Project layout

```
highway-yolo-agentic-demo/
├── app.py                  # Streamlit dashboard
├── requirements.txt
├── .gitignore
├── src/
│   ├── config.py           # All tunable constants
│   ├── tracking_utils.py   # Centroid tracker + counting-line helpers
│   ├── event_detection.py  # Wrong-way / stopped / speeding / congestion
│   ├── highway_monitor.py  # Orchestration pipeline (detection → tracking → events)
│   └── detect_video.py     # CLI entry-point
├── tests/
│   └── test_basic.py       # pytest unit tests (no GPU required)
├── samples/
│   └── README.md           # Where to place video files
└── outputs/
    └── .gitkeep            # Reports and annotated video land here
```

---

## Requirements

- Python 3.10+
- See `requirements.txt`

```bash
pip install -r requirements.txt
```

> YOLO11n weights (`yolo11n.pt`) are downloaded automatically by Ultralytics
> on the first run.

---

## Quick-start: CLI

```bash
python src/detect_video.py \
    --video samples/highway.mp4 \
    --output outputs \
    --conf 0.35 \
    --direction left_to_right \
    --save-video
```

### CLI flags

| Flag | Default | Description |
|---|---|---|
| `--video` | *(required)* | Path to input video (or `0` for webcam) |
| `--output` | `outputs` | Directory for CSV and video output |
| `--model` | `yolo11n.pt` | YOLO model weights path |
| `--conf` | `0.35` | Detection confidence threshold |
| `--iou` | `0.45` | NMS IoU threshold |
| `--imgsz` | `640` | Inference image size |
| `--direction` | `left_to_right` | Expected traffic direction |
| `--save-video` | off | Write annotated `.mp4` to `--output` |
| `--show` | off | Display live window (requires display) |
| `--max-frames` | `0` (all) | Stop after N frames |

**Valid directions:** `left_to_right`, `right_to_left`, `top_to_bottom`, `bottom_to_top`

---

## Quick-start: Streamlit dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser, upload a video from the sidebar,
configure parameters, and click **Run analysis**.

---

## Running tests

```bash
pytest tests/ -v
```

Tests cover:

- Configuration constants sanity checks
- `Track` state management and velocity computation
- `CentroidTracker` — assignment, ID stability, disappearance
- Counting-line geometry helpers
- `EventDetector` — wrong-way, stopped, speeding, congestion
- CLI argument parser

---

## Configuration

All tunable parameters live in `src/config.py`:

```python
DEFAULT_CONF = 0.35           # detection confidence
STOPPED_SPEED_THRESHOLD = 3.0 # px/frame below which a vehicle is "stopped"
STOPPED_DURATION_FRAMES = 60  # frames before a "stopped" event fires
SPEEDING_THRESHOLD_PX = 25.0  # px/frame above which a vehicle is "speeding"
CONGESTION_DENSITY_THRESHOLD = 8  # vehicles on screen at once → congestion
WRONG_WAY_CONSISTENCY_FRAMES = 10 # consecutive wrong-way frames before alert
COUNT_LINE_POSITION = 0.5     # 0–1 normalised position of the counting line
```

---

## Output files

After a run, `outputs/` contains:

| File | Contents |
|---|---|
| `events.csv` | One row per event: type, track ID, frame, timestamp, centroid, extras |
| `counts.csv` | Vehicles counted in each direction |
| `highway_output.mp4` | Annotated video (only with `--save-video`) |

---

## Architecture overview

```
Video frame
    │
    ▼
YOLO11n detect (vehicle classes only)
    │  list[(x1,y1,x2,y2,class_id)]
    ▼
CentroidTracker.update()
    │  dict[track_id → Track]
    ▼
┌───────────────────┬─────────────────────┐
│ Counting line     │ EventDetector        │
│ check_line_cross  │ wrong-way / stopped  │
│ counts["in/out"]  │ speeding / congestion│
└───────────────────┴─────────────────────┘
    │
    ▼
Annotate frame + append events
    │
    ▼
Display / write video / save CSV
```
