# scripts/generate_retargeting_demo.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from retargeting.hand_to_panda import HandToPandaRetargeter


# ---------------------------------------------------------------------------
# Minimal mock observation that mimics HandObservation
# ---------------------------------------------------------------------------
@dataclass
class MockHandObservation:
    landmarks_image: np.ndarray      # (21, 3) normalized image coords
    landmarks_world: np.ndarray      # (21, 3) fallback; same data here


def make_mock_hand(init_wrist_xy: tuple = (0.5, 0.5),
                   palm_span: float = 0.12) -> np.ndarray:
    """Return a (21, 3) landmark array in normalized image coordinates.

    Only wrist (idx 0) and middle MCP (idx 9) meaningfully affect the
    position mapping; the rest are filled with plausible neighbours so
    the downstream code does not crash on shape or NaN checks.
    """
    P = np.zeros((21, 3), dtype=np.float64)
    wx, wy = init_wrist_xy
    # wrist
    P[0] = [wx, wy, 0.0]
    # index MCP (5)  – slightly to the right, up from wrist
    P[5] = [wx + 0.04, wy - 0.03, 0.0]
    # middle MCP (9) – further up
    P[9] = [wx + 0.02, wy - palm_span, 0.0]
    # pinky MCP (17) – to the left, up
    P[17] = [wx - 0.03, wy - 0.02, 0.0]
    # thumb tip (4) and index tip (8) for pinch
    P[4] = [wx + 0.06, wy + 0.01, 0.0]
    P[8] = [wx + 0.06, wy - 0.05, 0.0]
    # fill remaining with wrist
    for i in range(21):
        if np.allclose(P[i], 0.0) and i != 0:
            P[i] = P[0] + [0.0, -0.01 * (i % 5), 0.0]
    return P


def generate_mock_trajectory(duration: float = 6.0,
                              fps: float = 30,
                              freq_xy: float = 0.35,
                              wrist_center: tuple = (0.5, 0.5),
                              amplitude_x: float = 0.18,
                              amplitude_y: float = 0.10,
                              pinch_cycle: float = 0.5) -> list:
    """Yield a sequence of MockHandObservation at *fps* for *duration* s."""
    n = int(duration * fps)
    out = []
    for i in range(n):
        t = i / fps
        # wrist moves in Lissajous-like pattern
        dx = amplitude_x * np.sin(2 * np.pi * freq_xy * t)
        dy = amplitude_y * np.sin(2 * np.pi * freq_xy * 0.7 * t)
        P = make_mock_hand(
            init_wrist_xy=(wrist_center[0] + dx, wrist_center[1] + dy),
            palm_span=0.12,
        )
        # modulate pinch over time
        angle = 2 * np.pi * pinch_cycle * t
        pinch_open = 0.5 * (1.0 + np.sin(angle))
        P[4, 0] = P[0, 0] + 0.06 + 0.03 * pinch_open
        P[8, 0] = P[0, 0] + 0.06 + 0.03 * pinch_open
        out.append(MockHandObservation(landmarks_image=P.copy(),
                                        landmarks_world=P.copy()))
    return out


def main():
    out_dir = Path("results/retargeting")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating mock hand trajectory (6 s @ 30 fps)…")

    # ------------------------------------------------------------------
    # Retargeter
    # ------------------------------------------------------------------
    retargeter = HandToPandaRetargeter(
        robot_origin=np.array([0.45, 0.0, 0.45]),
        position_scale_xy=2.2,
        depth_scale=0.0,
        filter_alpha=0.18,
    )

    frames = generate_mock_trajectory()
    rows = []

    for step, obs in enumerate(frames):
        target = retargeter.update(obs)
        rows.append({
            "step": step,
            "t": step / 30.0,
            "target_x": float(target.pos[0]),
            "target_y": float(target.pos[1]),
            "target_z": float(target.pos[2]),
            "gripper_width": float(target.gripper_width),
            "pinch_ratio": float(target.pinch_ratio),
            "valid": int(target.valid),
        })

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    csv_path = out_dir / "mock_hand_retargeting.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"Saved: {csv_path}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    t = np.array([r["t"] for r in rows])
    tx = np.array([r["target_x"] for r in rows])
    ty = np.array([r["target_y"] for r in rows])
    tz = np.array([r["target_z"] for r in rows])
    gw = np.array([r["gripper_width"] for r in rows])
    pr = np.array([r["pinch_ratio"] for r in rows])

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Top: position
    ax = axes[0]
    ax.plot(t, tx, label="target_x")
    ax.plot(t, ty, label="target_y")
    ax.plot(t, tz, label="target_z")
    ax.set_ylabel("Position [m]")
    ax.set_title("Mock Hand Retargeting – End-Effector Target Position")
    ax.legend()
    ax.grid(True)

    # Middle: gripper
    ax = axes[1]
    ax.plot(t, gw, label="gripper_width", color="green")
    ax.set_ylabel("Gripper width [m]")
    ax.set_title("Gripper Command")
    ax.legend()
    ax.grid(True)

    # Bottom: pinch ratio
    ax = axes[2]
    ax.plot(t, pr, label="pinch_ratio", color="purple")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Pinch ratio")
    ax.set_title("Pinch Ratio (input)")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    png_path = out_dir / "mock_hand_retargeting_curve.png"
    fig.savefig(png_path, dpi=200)
    print(f"Saved: {png_path}")
    plt.close(fig)

    print()
    print("Next: view the curve or include it in README / docs.")
    print("This offline demo does NOT require a webcam or MuJoCo viewer.")


if __name__ == "__main__":
    main()