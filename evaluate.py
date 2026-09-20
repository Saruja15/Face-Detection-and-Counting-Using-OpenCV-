"""Evaluate both detectors against your own hand-counted test images.

1. Put test images in  samples/
2. Create samples/ground_truth.csv with the true face count per image:

       filename,true_count,condition
       group1.jpg,6,well lit
       side_profile.jpg,2,side profile
       dark_room.jpg,3,low light

   (the 'condition' column is optional)
3. Run:  python evaluate.py

Outputs results/results.csv and prints a Markdown table you can paste into
the README's "Testing and Results" section.
"""
import csv
import statistics
import sys
from pathlib import Path

import cv2

from detector import HaarFaceDetector, YuNetFaceDetector

SAMPLES_DIR = Path("samples")
GROUND_TRUTH = SAMPLES_DIR / "ground_truth.csv"
RESULTS_DIR = Path("results")


def load_ground_truth():
    if not GROUND_TRUTH.exists():
        sys.exit(f"Missing {GROUND_TRUTH}. See the docstring at the top of evaluate.py.")
    with open(GROUND_TRUTH, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    rows = load_ground_truth()
    detectors = {"Haar Cascade": HaarFaceDetector(), "YuNet (DNN)": YuNetFaceDetector()}
    records = []

    for row in rows:
        path = SAMPLES_DIR / row["filename"]
        image = cv2.imread(str(path))
        if image is None:
            print(f"Skipping unreadable image: {path}")
            continue
        truth = int(row["true_count"])
        record = {"filename": row["filename"], "condition": row.get("condition", ""),
                  "true_count": truth}
        for name, det in detectors.items():
            res = det.detect(image)
            key = "haar" if name.startswith("Haar") else "yunet"
            record[f"{key}_count"] = res.count
            record[f"{key}_ms"] = round(res.elapsed_ms, 1)
        records.append(record)

    if not records:
        sys.exit("No images were evaluated.")

    RESULTS_DIR.mkdir(exist_ok=True)
    out_csv = RESULTS_DIR / "results.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    # ---- per-image Markdown table ---------------------------------------- #
    print("\n| Image | Condition | True | Haar | YuNet |")
    print("|---|---|---:|---:|---:|")
    for r in records:
        print(f"| {r['filename']} | {r['condition']} | {r['true_count']} | "
              f"{r['haar_count']} | {r['yunet_count']} |")

    # ---- summary ---------------------------------------------------------- #
    print("\n| Metric | Haar Cascade | YuNet (DNN) |")
    print("|---|---:|---:|")
    summary = {}
    for key, label in (("haar", "Haar Cascade"), ("yunet", "YuNet (DNN)")):
        errors = [abs(r[f"{key}_count"] - r["true_count"]) for r in records]
        exact = sum(e == 0 for e in errors) / len(records) * 100
        summary[label] = {
            "exact": f"{exact:.1f}%",
            "mae": f"{statistics.mean(errors):.2f}",
            "ms": f"{statistics.mean(r[f'{key}_ms'] for r in records):.1f}",
        }
    print(f"| Exact-count accuracy | {summary['Haar Cascade']['exact']} | {summary['YuNet (DNN)']['exact']} |")
    print(f"| Mean absolute count error | {summary['Haar Cascade']['mae']} | {summary['YuNet (DNN)']['mae']} |")
    print(f"| Avg. time per image (ms) | {summary['Haar Cascade']['ms']} | {summary['YuNet (DNN)']['ms']} |")
    print(f"\nImages evaluated: {len(records)}  |  Saved: {out_csv}")


if __name__ == "__main__":
    main()
