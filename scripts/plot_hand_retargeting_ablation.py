# scripts/plot_hand_retargeting_ablation.py

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


MODES_DISPLAY = {
    "full_pipeline": "Full pipeline",
    "no_smoothing": "No smoothing",
    "no_workspace_clamp": "No workspace clamp",
    "no_rate_limit": "No rate limit",
    "no_dropout_hold": "No dropout hold",
    "no_orientation_mapping": "No orientation",
    "no_pinch_gripper": "No pinch grip",
}

MODE_COLORS = {
    "full_pipeline": "#348ABD",
    "no_smoothing": "#E24A33",
    "no_workspace_clamp": "#988ED5",
    "no_rate_limit": "#F5A623",
    "no_dropout_hold": "#8EBA42",
    "no_orientation_mapping": "#46B3A0",
    "no_pinch_gripper": "#7F7F7F",
}


def load_per_frame(path: str):
    """Load a per-frame ablation CSV, return structured dict."""
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None,
                         encoding="utf-8", invalid_raise=False)
    out = {}
    if data is None or len(data) == 0:
        return None
    for name in data.dtype.names:
        col = data[name]
        # Convert empty strings / bytes to NaN for numeric columns
        if col.dtype.kind in ("U", "S", "O"):
            numeric = np.full(len(col), np.nan, dtype=np.float64)
            for i, val in enumerate(col):
                try:
                    numeric[i] = float(val)
                except (ValueError, TypeError):
                    numeric[i] = np.nan
            out[name] = numeric
        else:
            out[name] = col.astype(np.float64)
    return out


def load_metrics(path: str):
    """Load metrics CSV into {mode: {metric: value}}."""
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None,
                         encoding="utf-8", deletechars="", invalid_raise=False)
    out = {}
    if data is None or len(data) == 0:
        return out
    for row in data:
        mode = str(row[0])
        vals = {name: float(row[name]) for name in row.dtype.names[1:]}
        out[mode] = vals
    return out


# ---------------------------------------------------------------------------
# Individual plots
# ---------------------------------------------------------------------------

def plot_target_position(all_data, out_dir):
    """Target position over time for each mode."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    comps = ["target_pos_x", "target_pos_y", "target_pos_z"]
    labels = ["X [m]", "Y [m]", "Z [m]"]

    for mode_name, data in all_data.items():
        if data is None:
            continue
        t = data.get("timestamp", np.arange(len(data["target_pos_x"])) / 30.0)
        label = MODES_DISPLAY.get(mode_name, mode_name)
        color = MODE_COLORS.get(mode_name, None)
        for ax, comp, ylabel in zip(axes, comps, labels):
            ax.plot(t, data.get(comp, []), label=label, color=color,
                    linewidth=0.8, alpha=0.8)

    for ax, ylabel in zip(axes, labels):
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")

    axes[0].set_title("Target Position by Ablation Mode", fontsize=12)
    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    path = out_dir / "target_position_ablation.png"
    fig.savefig(path, dpi=200)
    print(f"Saved: {path}")
    plt.close(fig)


def plot_target_smoothness(all_data, out_dir):
    """Smoothness metrics: velocity, accel, jerk."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    comps = ["target_velocity_norm", "target_acceleration_norm",
              "target_jerk_norm"]
    ylabels = ["Velocity norm [m/s]", "Accel norm [m/s^2]", "Jerk norm [m/s^3]"]

    for mode_name, data in all_data.items():
        if data is None:
            continue
        t = data.get("timestamp", np.arange(100) / 30.0)
        label = MODES_DISPLAY.get(mode_name, mode_name)
        color = MODE_COLORS.get(mode_name, None)
        for ax, comp, ylabel in zip(axes, comps, ylabels):
            col = data.get(comp, np.full(len(t), np.nan))
            ax.plot(t, col, label=label, color=color, linewidth=0.8, alpha=0.8)

    for ax, ylabel in zip(axes, ylabels):
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")

    axes[0].set_title("Target Smoothness by Ablation Mode", fontsize=12)
    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    path = out_dir / "target_smoothness_ablation.png"
    fig.savefig(path, dpi=200)
    print(f"Saved: {path}")
    plt.close(fig)


def plot_ee_tracking_error(all_data, out_dir):
    """End-effector tracking error over time."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for mode_name, data in all_data.items():
        if data is None:
            continue
        t = data.get("timestamp", np.arange(100) / 30.0)
        label = MODES_DISPLAY.get(mode_name, mode_name)
        color = MODE_COLORS.get(mode_name, None)
        err = data.get("ee_position_error", np.full(len(t), np.nan))
        ax.plot(t, err, label=label, color=color, linewidth=0.8, alpha=0.8)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("EE position error [m]")
    ax.set_title("End-Effector Tracking Error by Ablation Mode")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "ee_tracking_error_ablation.png"
    fig.savefig(path, dpi=200)
    print(f"Saved: {path}")
    plt.close(fig)


def plot_torque_norm(all_data, out_dir):
    """Applied torque norm over time."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for mode_name, data in all_data.items():
        if data is None:
            continue
        t = data.get("timestamp", np.arange(100) / 30.0)
        label = MODES_DISPLAY.get(mode_name, mode_name)
        color = MODE_COLORS.get(mode_name, None)
        tau = data.get("torque_norm", np.full(len(t), np.nan))
        ax.plot(t, tau, label=label, color=color, linewidth=0.8, alpha=0.8)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Torque norm [Nm]")
    ax.set_title("Applied Torque Norm by Ablation Mode")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "torque_norm_ablation.png"
    fig.savefig(path, dpi=200)
    print(f"Saved: {path}")
    plt.close(fig)


def plot_gripper_command(all_data, out_dir):
    """Gripper width and pinch ratio over time."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    for ax, comp, ylabel, title in zip(
        axes,
        ["gripper_width", "pinch_ratio"],
        ["Gripper width [m]", "Pinch ratio"],
        ["Gripper Command by Ablation Mode", "Pinch Ratio (input)"],
    ):
        for mode_name, data in all_data.items():
            if data is None:
                continue
            t = data.get("timestamp", np.arange(100) / 30.0)
            label = MODES_DISPLAY.get(mode_name, mode_name)
            color = MODE_COLORS.get(mode_name, None)
            col = data.get(comp, np.full(len(t), np.nan))
            ax.plot(t, col, label=label, color=color, linewidth=0.8,
                    alpha=0.8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    path = out_dir / "gripper_command_ablation.png"
    fig.savefig(path, dpi=200)
    print(f"Saved: {path}")
    plt.close(fig)


def plot_summary_metrics(all_metrics, out_dir):
    """Separate bar charts for key metrics (no mixed axes)."""
    modes = [m for m in MODES_DISPLAY if m in all_metrics]
    n = len(modes)

    # Define 4 separate metric groups
    metric_groups = [
        ("End-effector position error [m]",
         ["mean_ee_position_error", "max_ee_position_error"],
         ["Mean", "Max"]),
        ("Target jerk RMS [m/s^3]",
         ["target_jerk_rms"],
         ["Jerk RMS"]),
        ("RMS torque [Nm]",
         ["rms_torque"],
         ["RMS torque"]),
        ("Workspace clip ratio",
         ["workspace_clip_ratio"],
         ["Clip ratio"]),
    ]

    n_groups = len(metric_groups)
    fig, axes = plt.subplots(1, n_groups, figsize=(5 * n_groups, 4.5))

    for gi, (title, keys, legends) in enumerate(metric_groups):
        ax = axes[gi]
        n_keys = len(keys)
        bar_width = 0.8 / n_keys
        x_pos = np.arange(n)

        for ki, key in enumerate(keys):
            values = [all_metrics[m].get(key, np.nan) for m in modes]
            colors = [MODE_COLORS.get(m, "#7F7F7F") for m in modes]
            offset = (ki - (n_keys - 1) / 2) * bar_width
            bars = ax.bar(x_pos + offset, values, bar_width,
                          color=colors, edgecolor="white", alpha=0.85,
                          label=legends[ki] if n_keys > 1 else None)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([MODES_DISPLAY.get(m, m) for m in modes],
                            rotation=20, ha="right", fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        if n_keys > 1:
            ax.legend(fontsize=7)

    fig.tight_layout()
    path = out_dir / "summary_metrics_ablation.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Plot hand retargeting ablation results")
    parser.add_argument("--data-dir", type=str,
                        default="results/retargeting/ablation")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    raw_dir = data_dir / "raw"
    figures_dir = data_dir / "figures"
    metrics_dir = data_dir / "metrics"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load per-frame data for each ablation mode
    all_data = {}
    mode_names = list(MODES_DISPLAY.keys())
    for mode in mode_names:
        csv_path = raw_dir / f"{mode}.csv"
        if not csv_path.exists():
            print(f"  [skip] {csv_path} not found")
            continue
        data = load_per_frame(str(csv_path))
        if data is not None:
            all_data[mode] = data
            print(f"  loaded: {mode} ({len(data.get('timestamp', []))} frames)")

    if not all_data:
        print("No data found. Run scripts/run_hand_retargeting_ablation.py first.")
        return

    # Load metrics
    metrics_path = metrics_dir / "hand_retargeting_ablation_metrics.csv"
    all_metrics = {}
    if metrics_path.exists():
        all_metrics = load_metrics(str(metrics_path))

    print(f"\nGenerating {6} plots from {len(all_data)} modes...")

    # Generate each plot
    plot_target_position(all_data, figures_dir)
    plot_target_smoothness(all_data, figures_dir)
    plot_ee_tracking_error(all_data, figures_dir)
    plot_torque_norm(all_data, figures_dir)
    plot_gripper_command(all_data, figures_dir)
    if all_metrics:
        plot_summary_metrics(all_metrics, figures_dir)

    print(f"\nAll plots saved to: {figures_dir}")


if __name__ == "__main__":
    main()
