# scripts/analyze_hand_retargeting_run.py
"""Analyze a recorded hand retargeting run and generate figures + metrics."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REQUIRED_COLS = ["timestamp", "frame_id", "detected_hand"]
POSITION_COLS = ["target_pos_x", "target_pos_y", "target_pos_z"]
ACTUAL_COLS = ["actual_ee_pos_x", "actual_ee_pos_y", "actual_ee_pos_z"]
GRIPPER_COLS = ["gripper_width", "pinch_ratio"]
TORQUE_COLS = ["torque_norm", "max_abs_torque"]


def load_run(path: str):
    """Load CSV and return dict of column_name -> numpy array."""
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None,
                         encoding="utf-8", invalid_raise=False)
    cols = {}
    if data is None or len(data) == 0:
        print("WARNING: empty or unreadable CSV")
        return cols
    for name in data.dtype.names:
        col = data[name]
        if col.dtype.kind in ("U", "S", "O"):
            numeric = np.full(len(col), np.nan, dtype=np.float64)
            for i, val in enumerate(col):
                try:
                    numeric[i] = float(val)
                except (ValueError, TypeError):
                    numeric[i] = np.nan
            cols[name] = numeric
        else:
            cols[name] = col.astype(np.float64)
    return cols


def compute_metrics(cols: dict, dt_avg: float) -> dict:
    """Compute metrics from loaded run data."""
    m = {}
    detected = cols.get("detected_hand", np.array([]))
    n = len(detected) if len(detected) > 0 else 1

    # 1. Detection rate
    m["detection_rate"] = float(np.mean(detected)) if len(detected) > 0 else np.nan

    # 2. EE position error
    ee_err = cols.get("ee_position_error", np.array([]))
    if len(ee_err) > 0:
        m["mean_ee_position_error"] = float(np.nanmean(ee_err))
        m["max_ee_position_error"] = float(np.nanmax(ee_err))
    else:
        m["mean_ee_position_error"] = m["max_ee_position_error"] = np.nan

    # 3. Target position smoothness (adjacent-frame diffs)
    sx, sy, sz = [cols.get(c, np.array([])) for c in ["target_pos_x", "target_pos_y", "target_pos_z"]]
    if len(sx) > 0:
        pos = np.column_stack([sx, sy, sz])
        diffs = np.diff(pos, axis=0)
        step_norms = np.linalg.norm(diffs, axis=1)
        m["target_position_smoothness"] = float(np.nanmean(step_norms))
        # Jump count: steps exceeding threshold
        threshold = 0.05  # 5 cm jump
        m["target_position_jump_count"] = int(np.sum(step_norms > threshold))
    else:
        m["target_position_smoothness"] = np.nan
        m["target_position_jump_count"] = np.nan

    # 4. Workspace clip rate
    clipped = cols.get("workspace_clipped", np.array([]))
    if len(clipped) > 0:
        m["workspace_clip_rate"] = float(np.mean(clipped > 0.5))
    else:
        m["workspace_clip_rate"] = np.nan

    # 5. Gripper smoothness
    gw = cols.get("gripper_width", np.array([]))
    if len(gw) > 0:
        gw_diff = np.abs(np.diff(gw))
        m["gripper_command_smoothness"] = float(np.nanmean(gw_diff))
        m["mean_gripper_width"] = float(np.nanmean(gw))
    else:
        m["gripper_command_smoothness"] = np.nan
        m["mean_gripper_width"] = np.nan

    # 6. Torque
    tn = cols.get("torque_norm", np.array([]))
    if len(tn) > 0:
        m["mean_torque_norm"] = float(np.nanmean(tn))
        m["max_torque_norm"] = float(np.nanmax(tn))
    else:
        m["mean_torque_norm"] = m["max_torque_norm"] = np.nan

    return m


def _plot_if_data(cols: dict, t: np.ndarray, figsize, out_dir, name, draw_func):
    """Call draw_func only if required data exists."""
    try:
        fig, ax = plt.subplots(figsize=figsize)
        draw_func(ax, cols, t)
        fig.tight_layout()
        path = out_dir / name
        fig.savefig(path, dpi=150)
        print(f"  Saved: {path}")
        plt.close(fig)
    except Exception as e:
        print(f"  [skip] {name}: {e}")


def plot_all(cols: dict, out_dir):
    """Generate all figures, skipping those with missing data."""
    out_dir.mkdir(parents=True, exist_ok=True)
    t = cols.get("timestamp", np.arange(len(next(iter(cols.values()))))
                 if cols else np.array([]))

    # 1. Target vs actual EE position
    def draw_target_vs_actual(ax, c, t):
        for comp, label, color in [("target_pos_x", "target_x", "blue"),
                                     ("target_pos_y", "target_y", "green"),
                                     ("target_pos_z", "target_z", "red")]:
            if comp in c:
                ax.plot(t, c[comp], label=label, color=color, linestyle="--", linewidth=1)
        for comp, label, color in [("actual_ee_pos_x", "actual_x", "blue"),
                                     ("actual_ee_pos_y", "actual_y", "green"),
                                     ("actual_ee_pos_z", "actual_z", "red")]:
            if comp in c:
                ax.plot(t, c[comp], label=label, color=color, linewidth=1, alpha=0.7)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Position [m]")
        ax.set_title("End-Effector Position: Target vs Actual")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    _plot_if_data(cols, t, (10, 5), out_dir, "target_vs_actual_ee_position.png",
                  draw_target_vs_actual)

    # 2. EE position error
    def draw_ee_error(ax, c, t):
        err = c.get("ee_position_error", np.array([]))
        if len(err) == 0:
            raise ValueError("No ee_position_error data")
        ax.plot(t, err, color="red", linewidth=1)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Position error [m]")
        ax.set_title("End-Effector Position Error")
        ax.grid(True, alpha=0.3)
    _plot_if_data(cols, t, (10, 4), out_dir, "ee_position_error.png", draw_ee_error)

    # 3. Gripper
    def draw_gripper(ax, c, t):
        gw = c.get("gripper_width", np.array([]))
        if len(gw) > 0:
            ax.plot(t, gw, label="gripper_width", color="green", linewidth=1)
        pr = c.get("pinch_ratio", np.array([]))
        if len(pr) > 0:
            ax.plot(t, pr, label="pinch_ratio", color="purple", linewidth=1, alpha=0.7)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Width / ratio")
        ax.set_title("Gripper Command and Pinch Ratio")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    _plot_if_data(cols, t, (10, 4), out_dir, "gripper_width_and_pinch.png", draw_gripper)

    # 4. Target smoothness
    def draw_smoothness(ax, c, t):
        sx, sy, sz = [c.get(n, np.array([])) for n in ["target_pos_x", "target_pos_y", "target_pos_z"]]
        if len(sx) == 0:
            raise ValueError("No target position data")
        pos = np.column_stack([sx, sy, sz])
        diffs = np.diff(pos, axis=0)
        step_norms = np.linalg.norm(diffs, axis=1)
        ax.plot(t[1:], step_norms, color="orange", linewidth=1)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Step norm [m]")
        ax.set_title("Target Position Smoothness (adjacent-frame diff)")
        ax.grid(True, alpha=0.3)
    _plot_if_data(cols, t, (10, 4), out_dir, "target_smoothness.png", draw_smoothness)

    # 5. Detection status
    def draw_detection(ax, c, t):
        det = c.get("detected_hand", np.array([]))
        if len(det) == 0:
            raise ValueError("No detection data")
        ax.fill_between(t, 0, det, color="green", alpha=0.5, label="detected")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Detected (0/1)")
        ax.set_title("Hand Detection Status")
        ax.set_ylim(-0.1, 1.1)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    _plot_if_data(cols, t, (10, 3), out_dir, "detection_status.png", draw_detection)

    # 6. Workspace clipping
    def draw_clipping(ax, c, t):
        clip = c.get("workspace_clipped", np.array([]))
        if len(clip) == 0 or np.all(np.isnan(clip)):
            raise ValueError("No workspace_clipped data")
        ax.fill_between(t, 0, clip, color="red", alpha=0.5, label="clipped")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Clipped (0/1)")
        ax.set_title("Workspace Clipping Events")
        ax.set_ylim(-0.1, 1.1)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    _plot_if_data(cols, t, (10, 3), out_dir, "workspace_clipping.png", draw_clipping)

    # 7. Torque norm
    def draw_torque(ax, c, t):
        tn = c.get("torque_norm", np.array([]))
        if len(tn) == 0:
            raise ValueError("No torque data")
        ax.plot(t, tn, color="purple", linewidth=1)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Torque norm [Nm]")
        ax.set_title("Control Torque Norm")
        ax.grid(True, alpha=0.3)
    _plot_if_data(cols, t, (10, 4), out_dir, "torque_norm.png", draw_torque)


def save_metrics(metrics: dict, out_dir):
    """Save metrics to CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "hand_retargeting_metrics.csv"
    with open(path, "w", encoding="utf-8") as f:
        f.write("metric,value\n")
        for k, v in metrics.items():
            val_str = f"{v:.6f}" if isinstance(v, float) else str(v)
            f.write(f"{k},{val_str}\n")
    print(f"  Saved: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Analyze hand retargeting run and generate figures + metrics")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to hand_retargeting_run.csv")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (defaults to run directory)")
    args = parser.parse_args()

    csv_path = Path(args.input)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        return

    run_dir = csv_path.parents[1] if args.output_dir is None else Path(args.output_dir)
    figures_dir = run_dir / "figures"
    metrics_dir = run_dir / "metrics"

    print(f"Loading: {csv_path}")
    cols = load_run(str(csv_path))
    if not cols:
        print("ERROR: could not read CSV")
        return

    # Estimate dt
    t = cols.get("timestamp", np.array([]))
    dt_avg = 1.0 / 30.0
    if len(t) > 1:
        dt_avg = float(np.median(np.diff(t)))

    print(f"  {len(t)} frames, dt ~ {dt_avg:.3f}s")

    # Compute metrics
    metrics = compute_metrics(cols, dt_avg)
    print(f"\nMetrics:")
    for k, v in metrics.items():
        val_str = f"{v:.6f}" if isinstance(v, float) else str(v)
        print(f"  {k}: {val_str}")

    # Generate figures
    print(f"\nGenerating figures:")
    plot_all(cols, figures_dir)

    # Save metrics
    save_metrics(metrics, metrics_dir)

    print(f"\nAnalysis complete. Results in: {run_dir}")


if __name__ == "__main__":
    main()
