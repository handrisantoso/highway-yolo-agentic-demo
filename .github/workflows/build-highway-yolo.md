---
on:
  workflow_dispatch:

engine: claude

permissions:
  contents: read
  issues: read
  pull-requests: read

safe-outputs:
  create-pull-request:
    allowed-files:
      - README.md
      - requirements.txt
      - .gitignore
      - src/**
      - tests/**
      - samples/**
      - outputs/**
      - app.py
  add-comment:
  create-issue:

tools:
  edit: {}
  bash: true
  github: {}
---

# Build YOLO11n Highway Monitoring System

You are Claude acting as a senior computer vision engineer.

Build a complete Python project in the current repository root.

Create a YOLO11n highway monitoring solution with:

- object detection
- vehicle counting
- wrong-way detection
- abnormal event detection
- Streamlit dashboard

Required files:

- README.md
- requirements.txt
- .gitignore
- src/detect_video.py
- src/highway_monitor.py
- src/tracking_utils.py
- src/event_detection.py
- src/config.py
- app.py
- tests/test_basic.py
- samples/README.md
- outputs/.gitkeep

Use:

- Python 3.10+
- ultralytics
- opencv-python
- numpy
- pandas
- streamlit
- pytest

Model:

- yolo11n.pt

CLI:

python src/detect_video.py --video samples/highway.mp4 --output outputs --conf 0.35 --direction left_to_right --save-video

Create pull request title:

Build YOLO11n highway monitoring system