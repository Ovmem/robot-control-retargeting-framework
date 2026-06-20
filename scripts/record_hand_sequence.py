# scripts/record_hand_sequence.py

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from vision.hand_tracker import MediaPipeHandTracker


LANDMARK_COLS = []
for i in range(21):
    LANDMARK_COLS += [f"l{i}_x", f"l{i}_y", f"l{i}_z"]
    LANDMARK_COLS += [f"wl{i}_x", f"wl{i}_y", f"wl{i}_z"]

FIELD_NAMES = ["timestamp", "step", "detected", "handedness", "score"] + LANDMARK_COLS


def record_sequence(camera_id: int, duration: float, output_path: str):
    """Record MediaPipe hand landmarks from webcam to CSV."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Opening camera {camera_id}...")
    tracker = MediaPipeHandTracker(
        camera_id=camera_id,
        draw=True,
        mirror=True,
    )

    fh = open(out_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=FIELD_NAMES)
    writer.writeheader()

    print(f"\nRecording for {duration}s. Press ESC in MediaPipe window to stop early.\n")
    start_time = time.time()
    step = 0

    try:
        while True:
            elapsed = time.time() - start_time
            if elapsed > duration:
                print(f"Reached {duration}s, stopping.")
                break

            obs = tracker.read()
            row = {
                "timestamp": f"{elapsed:.3f}",
                "step": step,
                "detected": "0",
                "handedness": "",
                "score": "",
            }

            if obs is not None:
                row["detected"] = "1"
                row["handedness"] = obs.handedness
                row["score"] = f"{obs.score:.4f}"

                P_img = obs.landmarks_image  # (21, 3)
                P_world = obs.landmarks_world  # (21, 3) or None

                for i in range(21):
                    row[f"l{i}_x"] = f"{P_img[i, 0]:.6f}"
                    row[f"l{i}_y"] = f"{P_img[i, 1]:.6f}"
                    row[f"l{i}_z"] = f"{P_img[i, 2]:.6f}"
                    if P_world is not None:
                        row[f"wl{i}_x"] = f"{P_world[i, 0]:.6f}"
                        row[f"wl{i}_y"] = f"{P_world[i, 1]:.6f}"
                        row[f"wl{i}_z"] = f"{P_world[i, 2]:.6f}"
                    else:
                        row[f"wl{i}_x"] = row[f"wl{i}_y"] = row[f"wl{i}_z"] = ""

            writer.writerow(row)
            step += 1

            # Check for ESC
            if cv2.waitKey(1) & 0xFF == 27:
                print("ESC pressed, stopping.")
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        fh.close()
        tracker.close()

    print(f"Saved: {out_path}  ({step} frames)")


def main():
    parser = argparse.ArgumentParser(
        description="Record hand landmark sequence from webcam")
    parser.add_argument("--camera-id", type=int, default=0,
                        help="Camera device index")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Recording duration in seconds")
    parser.add_argument("--output", type=str,
                        default="results/retargeting/camera/raw/hand_sequence.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    record_sequence(
        camera_id=args.camera_id,
        duration=args.duration,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
