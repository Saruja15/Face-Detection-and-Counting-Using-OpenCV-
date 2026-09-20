"""Streamlit web app: Face Detection and Counting Using OpenCV.

Run locally:   streamlit run app.py
"""
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from detector import (
    DETECTOR_NAMES,
    HaarFaceDetector,
    YuNetFaceDetector,
    annotate,
    resize_max_side,
    process_video,
)

st.set_page_config(page_title="Face Detection & Counting", page_icon="🙂", layout="wide")


# --------------------------------------------------------------------------- #
# Cached detector construction
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading detector...")
def load_detector(name: str):
    if name.startswith("YuNet"):
        return YuNetFaceDetector()
    return HaarFaceDetector()


def configure(detector, name: str, threshold: float, min_neighbors: int):
    if name.startswith("YuNet"):
        detector.set_score_threshold(threshold)
    else:
        detector.min_neighbors = min_neighbors
    return detector


def to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.title("Face Detection and Counting Using OpenCV")
st.caption("Upload an image or a video and the app detects and counts the faces in it.")

with st.sidebar:
    st.header("Settings")
    detector_name = st.selectbox("Detector", DETECTOR_NAMES)
    if detector_name.startswith("YuNet"):
        threshold = st.slider("Confidence threshold", 0.1, 0.99, 0.7, 0.01)
        min_neighbors = 5
    else:
        min_neighbors = st.slider("minNeighbors (higher = fewer false positives)", 1, 12, 5)
        threshold = 0.7
    show_scores = st.checkbox("Show confidence scores", value=False)

try:
    detector = configure(load_detector(detector_name), detector_name, threshold, min_neighbors)
except RuntimeError as err:
    st.error(str(err))
    st.stop()

tab_image, tab_video = st.tabs(["📷 Image", "🎞️ Video"])

# --------------------------------------------------------------------------- #
# Image tab
# --------------------------------------------------------------------------- #
with tab_image:
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded is not None:
        data = np.frombuffer(uploaded.read(), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            st.error("Could not read that file as an image.")
        else:
            image = resize_max_side(image)
            result = detector.detect(image)
            annotated = annotate(image, result, show_scores=show_scores)

            c1, c2, c3 = st.columns(3)
            c1.metric("Faces detected", result.count)
            c2.metric("Detector", detector_name)
            c3.metric("Inference time", f"{result.elapsed_ms:.1f} ms")

            left, right = st.columns(2)
            left.subheader("Original")
            left.image(to_rgb(image), use_container_width=True)
            right.subheader("Detected")
            right.image(to_rgb(annotated), use_container_width=True)

            ok, png = cv2.imencode(".png", annotated)
            if ok:
                st.download_button("Download annotated image", png.tobytes(),
                                   file_name="faces_detected.png", mime="image/png")

# --------------------------------------------------------------------------- #
# Video tab
# --------------------------------------------------------------------------- #
with tab_video:
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    stride = st.slider("Process every N-th frame (higher = faster)", 1, 10, 2)
    if video_file is not None and st.button("Process video"):
        suffix = Path(video_file.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
            tmp_in.write(video_file.read())
            in_path = tmp_in.name
        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

        bar = st.progress(0.0, text="Processing video...")
        try:
            stats = process_video(in_path, out_path, detector, frame_stride=stride,
                                  progress=lambda p: bar.progress(p, text="Processing video..."))
        except IOError as err:
            st.error(str(err))
            st.stop()
        bar.empty()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Frames read", stats.frames_read)
        c2.metric("Average faces / frame", f"{stats.avg_count:.2f}")
        c3.metric("Max faces in a frame", stats.max_count)
        c4.metric("Processing speed", f"{stats.avg_fps:.1f} FPS")

        st.subheader("Faces per processed frame")
        st.line_chart(stats.counts)

        with open(out_path, "rb") as fh:
            st.download_button("Download annotated video (MP4)", fh.read(),
                               file_name="faces_detected.mp4", mime="video/mp4")
        st.caption("Tip: some browsers cannot preview OpenCV's MP4 codec, so download the file to play it.")

st.divider()
st.caption("Built with OpenCV, NumPy and Streamlit.")
