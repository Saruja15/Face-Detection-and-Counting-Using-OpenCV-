"""Command-line interface for face detection and counting.

Examples
--------
    python main.py --source samples/group.jpg --output results/group_out.jpg
    python main.py --source samples/clip.mp4 --output results/clip_out.mp4
    python main.py --source 0 --detector haar          # live webcam (press q to quit)
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

from detector import annotate, get_detector, process_video

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Face detection and counting with OpenCV")
    p.add_argument("--source", required=True,
                   help="Path to an image/video, or a webcam index such as 0")
    p.add_argument("--detector", default="yunet", choices=["yunet", "haar"],
                   help="Detection method (default: yunet)")
    p.add_argument("--output", help="Where to save the annotated image/video")
    p.add_argument("--threshold", type=float, default=0.7,
                   help="YuNet confidence threshold (default: 0.7)")
    p.add_argument("--no-display", action="store_true", help="Do not open preview windows")
    return p.parse_args()


def build_detector(name: str, threshold: float):
    if name == "yunet":
        return get_detector("yunet", score_threshold=threshold)
    return get_detector("haar")


def run_image(path: str, detector, output, display: bool) -> None:
    image = cv2.imread(path)
    if image is None:
        sys.exit(f"Could not read image: {path}")
    result = detector.detect(image)
    annotated = annotate(image, result)
    print(f"Faces detected: {result.count}  ({result.elapsed_ms:.1f} ms)")
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output, annotated)
        print(f"Saved: {output}")
    if display:
        cv2.imshow("Face Detection", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_video(path: str, detector, output) -> None:
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
    stats = process_video(path, output, detector)
    print(f"Frames: {stats.frames_read} | avg faces/frame: {stats.avg_count:.2f} | "
          f"max: {stats.max_count} | speed: {stats.avg_fps:.1f} FPS")
    if output:
        print(f"Saved: {output}")


def run_webcam(index: int, detector, output) -> None:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        sys.exit(f"Cannot open webcam {index}")
    writer = None
    prev = time.perf_counter()
    print("Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = detector.detect(frame)
        annotated = annotate(frame, result)

        now = time.perf_counter()
        fps = 1.0 / max(now - prev, 1e-9)
        prev = now
        cv2.putText(annotated, f"{fps:.1f} FPS", (10, annotated.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        if output and writer is None:
            h, w = annotated.shape[:2]
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h))
        if writer is not None:
            writer.write(annotated)

        cv2.imshow("Face Detection (q to quit)", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    detector = build_detector(args.detector, args.threshold)
    display = not args.no_display

    if args.source.isdigit():
        run_webcam(int(args.source), detector, args.output)
    elif Path(args.source).suffix.lower() in IMAGE_EXTS:
        run_image(args.source, detector, args.output, display)
    else:
        run_video(args.source, detector, args.output)


if __name__ == "__main__":
    main()
