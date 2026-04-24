"""
Streamlit dashboard for YOLO11n highway monitoring.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import (
    DEFAULT_CONF,
    DEFAULT_IOU,
    DEFAULT_IMGSZ,
    MODEL_NAME,
    VALID_DIRECTIONS,
)
from highway_monitor import HighwayMonitor


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Highway YOLO11n Monitor",
    page_icon="🛣️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "monitor" not in st.session_state:
    st.session_state.monitor = None
if "events_df" not in st.session_state:
    st.session_state.events_df = pd.DataFrame()
if "counts" not in st.session_state:
    st.session_state.counts = {"in": 0, "out": 0}
if "processing" not in st.session_state:
    st.session_state.processing = False


# ---------------------------------------------------------------------------
# Sidebar – configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Configuration")

    uploaded_video = st.file_uploader(
        "Upload highway video",
        type=["mp4", "avi", "mov", "mkv"],
        help="Supported formats: MP4, AVI, MOV, MKV",
    )

    model_path = st.text_input(
        "Model weights",
        value=MODEL_NAME,
        help="Path to YOLO11n .pt file. 'yolo11n.pt' is auto-downloaded.",
    )

    direction = st.selectbox(
        "Traffic direction",
        options=VALID_DIRECTIONS,
        index=0,
        help="Expected direction of normal traffic flow.",
    )

    conf_threshold = st.slider(
        "Confidence threshold",
        min_value=0.10, max_value=0.95, value=DEFAULT_CONF, step=0.05,
    )

    iou_threshold = st.slider(
        "IoU threshold",
        min_value=0.10, max_value=0.95, value=DEFAULT_IOU, step=0.05,
    )

    imgsz = st.select_slider(
        "Inference image size",
        options=[320, 416, 512, 640, 768, 1024],
        value=DEFAULT_IMGSZ,
    )

    max_frames = st.number_input(
        "Max frames (0 = all)",
        min_value=0, max_value=10000, value=0, step=50,
    )

    save_csv = st.checkbox("Save CSV reports", value=True)


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("🛣️ Highway YOLO11n Monitoring System")
st.markdown(
    "Real-time vehicle detection, counting, wrong-way detection and "
    "abnormal event analysis powered by **YOLO11n**."
)

# ---------------------------------------------------------------------------
# Metrics row
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
metric_vehicles = col1.empty()
metric_in = col2.empty()
metric_out = col3.empty()
metric_events = col4.empty()

metric_vehicles.metric("Vehicles on screen", 0)
metric_in.metric("Counted IN", 0)
metric_out.metric("Counted OUT", 0)
metric_events.metric("Total events", 0)


# ---------------------------------------------------------------------------
# Video + event feed
# ---------------------------------------------------------------------------
col_video, col_events = st.columns([3, 2])

with col_video:
    st.subheader("Live feed")
    video_placeholder = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()

with col_events:
    st.subheader("Event log")
    events_placeholder = st.empty()

# ---------------------------------------------------------------------------
# Charts (below main row)
# ---------------------------------------------------------------------------
st.markdown("---")
chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.subheader("Event type distribution")
    chart_event_types = st.empty()
with chart_col2:
    st.subheader("Vehicles over time")
    chart_vehicles = st.empty()

vehicle_counts_over_time: list[int] = []

# ---------------------------------------------------------------------------
# Run button
# ---------------------------------------------------------------------------
run_btn = st.sidebar.button(
    "▶  Run analysis",
    disabled=uploaded_video is None,
    type="primary",
)

if uploaded_video is None:
    st.info("Upload a video using the sidebar to get started.")

# ---------------------------------------------------------------------------
# Processing loop
# ---------------------------------------------------------------------------
if run_btn and uploaded_video is not None:
    # Write upload to a temp file
    suffix = Path(uploaded_video.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_video.read())
        tmp_path = tmp.name

    output_dir = "outputs" if save_csv else None

    monitor = HighwayMonitor(
        model_path=model_path,
        direction=direction,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=imgsz,
        output_dir=output_dir,
        save_video=False,
    )

    cap = cv2.VideoCapture(tmp_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frame_idx = 0

    stop_btn_placeholder = st.sidebar.empty()
    stop = stop_btn_placeholder.button("⏹  Stop")

    all_events_records: list[dict] = []

    try:
        while cap.isOpened() and not stop:
            ret, frame = cap.read()
            if not ret:
                break

            annotated, new_events = monitor.process_frame(frame)
            frame_idx += 1
            vehicle_counts_over_time.append(
                len(monitor._tracker.tracks)
            )

            # Collect event records
            for ev in new_events:
                all_events_records.append(ev.to_dict())

            # Update video frame (convert BGR → RGB for Streamlit)
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb, channels="RGB", use_container_width=True)

            # Update progress
            pct = frame_idx / total
            progress_bar.progress(min(pct, 1.0))
            status_text.text(f"Frame {frame_idx}/{total}")

            # Update metrics
            metric_vehicles.metric("Vehicles on screen", len(monitor._tracker.tracks))
            metric_in.metric("Counted IN", monitor.counts["in"])
            metric_out.metric("Counted OUT", monitor.counts["out"])
            metric_events.metric("Total events", len(all_events_records))

            # Update event log (last 20)
            if all_events_records:
                df = pd.DataFrame(all_events_records[-20:])
                events_placeholder.dataframe(
                    df[["frame_number", "event_type", "track_id", "class_name", "extra"]],
                    use_container_width=True,
                    hide_index=True,
                )

            # Charts every 30 frames
            if frame_idx % 30 == 0:
                if all_events_records:
                    df_all = pd.DataFrame(all_events_records)
                    chart_event_types.bar_chart(
                        df_all["event_type"].value_counts()
                    )
                if len(vehicle_counts_over_time) > 1:
                    chart_vehicles.line_chart(vehicle_counts_over_time)

            if max_frames and frame_idx >= max_frames:
                break

    finally:
        cap.release()
        monitor.close()
        stop_btn_placeholder.empty()

    progress_bar.progress(1.0)
    status_text.text("Analysis complete.")

    # Final summary
    st.success(
        f"Processed {frame_idx} frames | "
        f"IN: {monitor.counts['in']}  OUT: {monitor.counts['out']} | "
        f"Events: {len(all_events_records)}"
    )

    # Full events table
    if all_events_records:
        st.markdown("---")
        st.subheader("All detected events")
        df_final = pd.DataFrame(all_events_records)
        st.dataframe(df_final, use_container_width=True, hide_index=True)

        # Download button
        csv_bytes = df_final.to_csv(index=False).encode()
        st.download_button(
            label="⬇ Download events CSV",
            data=csv_bytes,
            file_name="highway_events.csv",
            mime="text/csv",
        )

    # Final charts
    if all_events_records:
        df_all = pd.DataFrame(all_events_records)
        chart_event_types.bar_chart(df_all["event_type"].value_counts())
    if vehicle_counts_over_time:
        chart_vehicles.line_chart(vehicle_counts_over_time)
