# scripts/sweep_hand_retargeting_params.py
"""Replay recorded hand trajectory with different mapping parameters and compare metrics."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from retargeting.hand_to_panda import HandToPandaRetargeter


@dataclass
class MockObs:
    landmarks_image: np.ndarray
    landmarks_world: np.ndarray


def make_hand_from_wrist(wrist_xy, pinch_ratio_val, palm_span=0.12):
    """Create 21 landmarks from wrist position and pinch ratio."""
    wx, wy = wrist_xy
    P = np.zeros((21, 3), dtype=np.float64)
    P[0] = [wx, wy, 0.0]
    P[5] = [wx + 0.04, wy - 0.03, 0.0]
    P[9] = [wx + 0.02, wy - palm_span, 0.0]
    P[17] = [wx - 0.03, wy - 0.02, 0.0]
    spread = 0.02 + pinch_ratio_val * 0.03
    P[4] = [wx + 0.06 + spread, wy + 0.01, 0.0]
    P[8] = [wx + 0.06 - spread, wy - 0.05, 0.0]
    for i in range(21):
        if np.allclose(P[i], 0.0) and i != 0:
            P[i] = P[0] + [0.0, -0.01 * (i % 5), 0.0]
    return P


SWEEP_CONFIGS = [
    {"name": "default", "scale": 2.2, "alpha": 0.18, "bounds_y": 0.25, "bounds_z": [0.12, 0.14], "orientation": True, "gripper": True},
    {"name": "high_scale", "scale": 3.5, "alpha": 0.18, "bounds_y": 0.25, "bounds_z": [0.12, 0.14], "orientation": True, "gripper": True},
    {"name": "more_smoothing", "scale": 2.2, "alpha": 0.08, "bounds_y": 0.25, "bounds_z": [0.12, 0.14], "orientation": True, "gripper": True},
    {"name": "less_smoothing", "scale": 2.2, "alpha": 0.40, "bounds_y": 0.25, "bounds_z": [0.12, 0.14], "orientation": True, "gripper": True},
    {"name": "tight_bounds", "scale": 2.2, "alpha": 0.18, "bounds_y": 0.15, "bounds_z": [0.06, 0.08], "orientation": True, "gripper": True},
    {"name": "no_orientation", "scale": 2.2, "alpha": 0.18, "bounds_y": 0.25, "bounds_z": [0.12, 0.14], "orientation": False, "gripper": True},
    {"name": "no_gripper", "scale": 2.2, "alpha": 0.18, "bounds_y": 0.25, "bounds_z": [0.12, 0.14], "orientation": True, "gripper": False},
]


def run_sweep_config(config: dict, wrist_positions: np.ndarray,
                      pinch_values: np.ndarray, robot_origin: np.ndarray) -> dict:
    """Run retargeter with given config over recorded wrist/pinch data."""
    retargeter = HandToPandaRetargeter(
        robot_origin=robot_origin.copy(),
        position_scale_xy=config["scale"],
        filter_alpha=config["alpha"],
    )

    # Override bounds (approximate - retargeter's internal bounds are hardcoded)
    # For a proper sweep we'd need to make workspace bounds configurable in the retargeter
    # Here we approximate by checking if targets would be clipped

    smoothness_values = []
    clip_count = 0
    gripper_diffs = []
    prev_pos = None

    for i in range(len(wrist_positions)):
        wx, wy = wrist_positions[i]
        P = make_hand_from_wrist((wx, wy), pinch_values[i])
        obs = MockObs(landmarks_image=P, landmarks_world=P)
        target = retargeter.update(obs)

        if not target.valid:
            continue

        # Smoothness
        if prev_pos is not None:
            diff = np.linalg.norm(target.pos - prev_pos)
            smoothness_values.append(diff)
        prev_pos = target.pos.copy()

        # Approximate clipping
        y = target.pos[1]
        z = target.pos[2]
        if abs(y) > config["bounds_y"]:
            clip_count += 1
        elif z < robot_origin[2] - config["bounds_z"][0] or z > robot_origin[2] + config["bounds_z"][1]:
            clip_count += 1

        # Gripper smoothness
        if len(gripper_diffs) > 0:
            gripper_diffs.append(abs(target.gripper_width - gripper_diffs[-1]) if gripper_diffs else 0.0)
        else:
            gripper_diffs.append(0.0)

    smoothness = float(np.nanmean(smoothness_values)) if smoothness_values else np.nan
    clip_rate = clip_count / max(len(wrist_positions), 1)
    gripper_smooth = float(np.nanmean(np.abs(np.diff(gripper_diffs)))) if len(gripper_diffs) > 1 else np.nan

    return {
        "smoothness": smoothness,
        "workspace_clip_rate": clip_rate,
        "gripper_smoothness": gripper_smooth,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Sweep hand retargeting parameters over real recorded data")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to hand_retargeting_run.csv (real run data)")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    csv_path = Path(args.input)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        return

    run_dir = csv_path.parents[1] if args.output_dir is None else Path(args.output_dir)

    # Load wrist positions and pinch from real run
    wrist_positions = []
    pinch_values = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("detected_hand", "0") == "1":
                wx = float(row.get("wrist_x", 0.5))
                wy = float(row.get("wrist_y", 0.5))
                pr = float(row.get("pinch_ratio", 0.8))
                wrist_positions.append((wx, wy))
                pinch_values.append(min(max(pr, 0.1), 1.5))

    if len(wrist_positions) == 0:
        print("ERROR: no detected frames in run data")
        return

    print(f"Loaded {len(wrist_positions)} detected frames from {csv_path}")

    wrist_arr = np.array(wrist_positions)
    pinch_arr = np.array(pinch_values)
    robot_origin = np.array([0.45, 0.0, 0.45])

    # Run sweep
    results = []
    for cfg in SWEEP_CONFIGS:
        metrics = run_sweep_config(cfg, wrist_arr, pinch_arr, robot_origin)
        results.append({**cfg, **metrics})
        print(f"  {cfg['name']:20s} smooth={metrics['smoothness']:.6f} clip={metrics['workspace_clip_rate']:.3f}")

    # Save results
    sweep_dir = run_dir / "param_sweep"
    raw_dir = sweep_dir / "raw"
    figures_dir = sweep_dir / "figures"
    metrics_dir = sweep_dir / "metrics"
    raw_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Raw CSV
    fieldnames = ["name", "scale", "alpha", "bounds_y", "orientation", "gripper",
                   "smoothness", "workspace_clip_rate", "gripper_smoothness"]
    csv_out = raw_dir / "param_sweep_results.csv"
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"  Saved: {csv_out}")

    # Find best params (min smoothness, then clip)
    best = min(results, key=lambda r: (r.get("workspace_clip_rate", 1), r.get("smoothness", 1)))
    best_out = metrics_dir / "best_params.csv"
    with open(best_out, "w", encoding="utf-8") as f:
        f.write("param,value\n")
        for k, v in best.items():
            f.write(f"{k},{v}\n")
    print(f"  Best: {best_out} ({best['name']})")

    # Plot smoothness
    fig, ax = plt.subplots(figsize=(8, 4))
    names = [r["name"] for r in results]
    vals = [r["smoothness"] if r["smoothness"] is not None else 0 for r in results]
    colors = ["#348ABD", "#E24A33", "#988ED5", "#F5A623", "#8EBA42", "#46B3A0", "#7F7F7F"]
    ax.bar(names, vals, color=colors[:len(names)], edgecolor="white")
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Target smoothness [m]")
    ax.set_title("Parameter Sweep: Target Smoothness")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "param_sweep_smoothness.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {figures_dir / 'param_sweep_smoothness.png'}")

    # Plot clip rate
    fig, ax = plt.subplots(figsize=(8, 4))
    clip_vals = [r["workspace_clip_rate"] for r in results]
    ax.bar(names, clip_vals, color=colors[:len(names)], edgecolor="white")
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Workspace clip rate")
    ax.set_title("Parameter Sweep: Workspace Clip Rate")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "param_sweep_workspace_clip_rate.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {figures_dir / 'param_sweep_workspace_clip_rate.png'}")

    # Plot gripper smoothness
    fig, ax = plt.subplots(figsize=(8, 4))
    gs_vals = [r["gripper_smoothness"] if r["gripper_smoothness"] is not None else 0 for r in results]
    ax.bar(names, gs_vals, color=colors[:len(names)], edgecolor="white")
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Gripper smoothness")
    ax.set_title("Parameter Sweep: Gripper Command Smoothness")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "param_sweep_gripper_smoothness.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {figures_dir / 'param_sweep_gripper_smoothness.png'}")

    print(f"\nSweep complete. Results in: {sweep_dir}")
    print(f"Best params: {best['name']}")


if __name__ == "__main__":
    main()
