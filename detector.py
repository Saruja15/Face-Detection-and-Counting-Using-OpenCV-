"""Face detection and counting utilities built on OpenCV.

Two detectors share one interface:

* ``HaarFaceDetector``  - classic Viola-Jones Haar cascade (fast baseline).
* ``YuNetFaceDetector`` - deep-learning detector (OpenCV ``FaceDetectorYN``),
  noticeably more accurate on angled, small, or partly hidden faces.

Every detector exposes ``detect(image_bgr) -> DetectionResult``.
"""
from __future__ import annotations

import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]  # x, y, w, h

MODEL_DIR = Path(__file__).resolve().parent / "models"
YUNET_FILE = "face_detection_yunet_2023mar.onnx"
YUNET_PATH = MODEL_DIR / YUNET_FILE
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    f"face_detection_yunet/{YUNET_FILE}"
)

DETECTOR_NAMES = ("YuNet (DNN)", "Haar Cascade")


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass
class DetectionResult:
    """Output of a single detection pass."""

    boxes: List[Box] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def count(self) -> int:
        return len(self.boxes)


@dataclass
class VideoStats:
    """Summary of a processed video."""

    frames_read: int
    frames_processed: int
    counts: List[int]
    avg_count: float
    max_count: int
    avg_fps: float
    output_path: Optional[str]


# --------------------------------------------------------------------------- #
# Detectors
# --------------------------------------------------------------------------- #
class HaarFaceDetector:
    """Viola-Jones Haar cascade detector (ships with OpenCV)."""

    name = "Haar Cascade"

    def __init__(
        self,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (30, 30),
    ) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError(f"Could not load Haar cascade from {cascade_path}")
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def detect(self, image_bgr: np.ndarray) -> DetectionResult:
        start = time.perf_counter()
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # helps under uneven lighting
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )
        boxes = [tuple(int(v) for v in f) for f in faces]
        elapsed = (time.perf_counter() - start) * 1000.0
        # Haar cascades do not give a calibrated confidence score.
        return DetectionResult(boxes=boxes, scores=[1.0] * len(boxes), elapsed_ms=elapsed)


def ensure_yunet_model(path: Path = YUNET_PATH, url: str = YUNET_URL) -> Path:
    """Download the YuNet ONNX model on first use."""
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading YuNet model to {path} ...")
    try:
        urllib.request.urlretrieve(url, path)
    except Exception as exc:  # network problems, blocked host, etc.
        if path.exists():
            path.unlink()
        raise RuntimeError(
            "Could not download the YuNet model. Download it manually from\n"
            f"  {url}\nand place it in the 'models/' folder."
        ) from exc
    return path


class YuNetFaceDetector:
    """Deep-learning face detector (YuNet) via ``cv2.FaceDetectorYN``."""

    name = "YuNet (DNN)"

    def __init__(
        self,
        score_threshold: float = 0.7,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ) -> None:
        if not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError("OpenCV >= 4.5.4 is required for FaceDetectorYN.")
        model_path = ensure_yunet_model()
        self.detector = cv2.FaceDetectorYN.create(
            str(model_path), "", (320, 320), score_threshold, nms_threshold, top_k
        )

    def set_score_threshold(self, value: float) -> None:
        self.detector.setScoreThreshold(float(value))

    def detect(self, image_bgr: np.ndarray) -> DetectionResult:
        start = time.perf_counter()
        h, w = image_bgr.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(image_bgr)
        boxes: List[Box] = []
        scores: List[float] = []
        if faces is not None:
            for f in faces:
                x, y, bw, bh = (int(v) for v in f[:4])
                boxes.append((x, y, bw, bh))
                scores.append(float(f[-1]))
        elapsed = (time.perf_counter() - start) * 1000.0
        return DetectionResult(boxes=boxes, scores=scores, elapsed_ms=elapsed)


def get_detector(name: str, **kwargs):
    """Factory: return a detector by its display name."""
    key = name.lower()
    if key.startswith("yunet") or key in {"dnn", "yunet"}:
        return YuNetFaceDetector(**kwargs)
    if key.startswith("haar"):
        return HaarFaceDetector(**kwargs)
    raise ValueError(f"Unknown detector '{name}'. Choose from {DETECTOR_NAMES}.")


# --------------------------------------------------------------------------- #
# Drawing helpers
# --------------------------------------------------------------------------- #
def annotate(
    image_bgr: np.ndarray,
    result: DetectionResult,
    color: Tuple[int, int, int] = (0, 200, 0),
    show_scores: bool = False,
) -> np.ndarray:
    """Return a copy of the image with boxes and a face-count banner."""
    out = image_bgr.copy()
    thickness = max(2, int(round(min(out.shape[:2]) / 300)))
    for i, (x, y, w, h) in enumerate(result.boxes):
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
        if show_scores:
            label = f"{result.scores[i]:.2f}"
            cv2.putText(out, label, (x, max(15, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    banner = f"Faces: {result.count}"
    scale = max(0.7, min(out.shape[:2]) / 700)
    (tw, th), base = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.rectangle(out, (0, 0), (tw + 20, th + base + 16), (0, 0, 0), -1)
    cv2.putText(out, banner, (10, th + 8), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def resize_max_side(image_bgr: np.ndarray, max_side: int = 1280) -> np.ndarray:
    """Downscale very large images so detection stays fast."""
    h, w = image_bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image_bgr
    ratio = max_side / float(longest)
    return cv2.resize(image_bgr, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------- #
# Image / video processing
# --------------------------------------------------------------------------- #
def process_image(image_bgr: np.ndarray, detector) -> Tuple[np.ndarray, DetectionResult]:
    result = detector.detect(image_bgr)
    return annotate(image_bgr, result), result


def process_video(
    input_path: str,
    output_path: Optional[str],
    detector,
    frame_stride: int = 1,
    max_frames: Optional[int] = None,
    progress: Optional[Callable[[float], None]] = None,
) -> VideoStats:
    """Detect and count faces in every ``frame_stride``-th frame of a video.

    Frames that are skipped reuse the last annotation so the output video
    keeps its original length and frame rate.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {input_path}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps_in, (width, height))

    counts: List[int] = []
    frames_read = 0
    frames_processed = 0
    last_result = DetectionResult()
    start = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames_read += 1
        if (frames_read - 1) % max(1, frame_stride) == 0:
            last_result = detector.detect(frame)
            frames_processed += 1
            counts.append(last_result.count)
        if writer is not None:
            writer.write(annotate(frame, last_result))
        if progress and total:
            progress(min(1.0, frames_read / total))
        if max_frames and frames_read >= max_frames:
            break

    elapsed = max(time.perf_counter() - start, 1e-9)
    cap.release()
    if writer is not None:
        writer.release()

    return VideoStats(
        frames_read=frames_read,
        frames_processed=frames_processed,
        counts=counts,
        avg_count=float(np.mean(counts)) if counts else 0.0,
        max_count=int(max(counts)) if counts else 0,
        avg_fps=frames_read / elapsed,
        output_path=output_path,
    )
