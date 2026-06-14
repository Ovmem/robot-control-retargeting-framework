import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from vision.hand_tracker import MediaPipeHandTracker


def main():
    tracker = MediaPipeHandTracker(
        camera_id=0,
        max_num_hands=1,
        draw=True,
        mirror=True,
    )

    print("Press Ctrl+C to exit.")

    try:
        while True:
            obs = tracker.read()

            if obs is not None:
                print(
                    f"hand={obs.handedness}, "
                    f"score={obs.score:.2f}, "
                    f"image_shape={obs.landmarks_image.shape}, "
                    f"world={obs.landmarks_world is not None}"
                )

    except KeyboardInterrupt:
        pass
    finally:
        tracker.close()


if __name__ == "__main__":
    main()
