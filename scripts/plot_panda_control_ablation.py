# scripts/plot_panda_control_ablation.py

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


def read_ablation_csv(path):
    """Read an ablation CSV and return (t, err_norm, tau_norm)."""
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None,
                         encoding="utf-8")
    return data["t"], data["err_norm"], data["tau_norm"]


def read_metrics_csv(path):
    """Read metrics CSV and return {mode: {metric: value}}."""
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None,
                         encoding="utf-8", deletechars="")
    out = {}
    for row in data:
        mode = row[0]
        vals = {name: float(row[name]) for name in row.dtype.names[1:]}
        out[mode] = vals
    return out


MODES_DISPLAY = {
    "pd_only": "PD only",
    "pd_gc": "PD + GC",
    "pd_gc_low_gain": "PD + GC (low gain)",
    "pd_gc_torque_clipped": "PD + GC (clipped)",
    "computed_torque": "Computed torque",
    "task_space_pd_gc": "Task-space PD + GC",
}

MODE_COLORS = {
    "pd_only": "#E24A33",
    "pd_gc": "#348ABD",
    "pd_gc_low_gain": "#988ED5",
    "pd_gc_torque_clipped": "#F5A623",
    "computed_torque": "#8EBA42",
    "task_space_pd_gc": "#46B3A0",
}


def plot_error_over_time(all_data, out_dir):
    """Joint error norm over time for each ablation mode."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for mode, (t, err, _) in all_data.items():
        label = MODES_DISPLAY.get(mode, mode)
        color = MODE_COLORS.get(mode, None)
        ax.plot(t, err, label=label, color=color, linewidth=1.2)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Joint error norm [rad]")
    ax.set_title("Joint Tracking Error by Control Mode")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "control_ablation_error.png"
    fig.savefig(path, dpi=200)
    print(f"Saved: {path}")
    plt.close(fig)


def plot_torque_over_time(all_data, out_dir):
    """Torque norm over time for each ablation mode."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for mode, (t, _, tau) in all_data.items():
        label = MODES_DISPLAY.get(mode, mode)
        color = MODE_COLORS.get(mode, None)
        ax.plot(t, tau, label=label, color=color, linewidth=1.2)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Torque norm [Nm]")
    ax.set_title("Applied Torque Norm by Control Mode")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "control_ablation_torque.png"
    fig.savefig(path, dpi=200)
    print(f"Saved: {path}")
    plt.close(fig)


def plot_summary_bar(all_metrics, out_dir):
    """Grouped bar chart of key metrics across modes."""
    modes = list(all_metrics.keys())
    n = len(modes)

    metric_keys = [
        ("mean_joint_error", "Mean joint error [rad]"),
        ("final_joint_error", "Final joint error [rad]"),
        ("rms_torque", "RMS torque [Nm]"),
        ("max_torque", "Max torque [Nm]"),
        ("torque_smoothness", "Torque smoothness"),
    ]
    n_metrics = len(metric_keys)

    fig, axes = plt.subplots(1, n_metrics, figsize=(3.2 * n_metrics, 4.5),
                              sharey=False)

    for ax_idx, (key, ylabel) in enumerate(metric_keys):
        ax = axes[ax_idx]
        values = [all_metrics[m].get(key, float("nan")) for m in modes]
        colors = [MODE_COLORS.get(m, "#7F7F7F") for m in modes]
        x_pos = np.arange(n)
        bars = ax.bar(x_pos, values, color=colors, width=0.6, edgecolor="white")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([MODES_DISPLAY.get(m, m) for m in modes],
                           rotation=25, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(key.replace("_", " ").title(), fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)

        for bar, val in zip(bars, values):
            if not np.isnan(val):
                fmt = ".2f" if val < 100 else ".1f"
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f"{val:{fmt}}",
                        ha="center", va="bottom", fontsize=6.5,
                        rotation=45)

    fig.tight_layout()
    path = out_dir / "control_ablation_summary.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot Panda control ablation results")
    parser.add_argument("--data-dir", type=str,
                        default="results/panda_control")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    raw_dir = data_dir / "raw"
    figures_dir = data_dir / "figures"
    metrics_dir = data_dir / "metrics"
    figures_dir.mkdir(parents=True, exist_ok=True)

    mode_names = list(MODES_DISPLAY.keys())
    all_data = {}
    missing = []
    for mode in mode_names:
        csv_path = raw_dir / f"ablation_{mode}.csv"
        if not csv_path.exists():
            missing.append(str(csv_path))
            continue
        t, err, tau = read_ablation_csv(csv_path)
        all_data[mode] = (t, err, tau)

    if missing:
        print("WARNING: missing CSV files, those modes will be skipped:")
        for p in missing:
            print(f"  {p}")
        print("Run scripts/run_panda_control_ablation.py first.")

    if not all_data:
        print("No data to plot. Exiting.")
        return

    metrics_path = metrics_dir / "control_ablation_metrics.csv"
    if metrics_path.exists():
        all_metrics = read_metrics_csv(metrics_path)
    else:
        print("WARNING: metrics CSV not found; summary chart will be skipped.")
        all_metrics = {}

    print(f"Generating plots from {len(all_data)} modes...")
    plot_error_over_time(all_data, figures_dir)
    plot_torque_over_time(all_data, figures_dir)
    if all_metrics:
        plot_summary_bar(all_metrics, figures_dir)

    print(f"\nAll plots saved to: {figures_dir}")


if __name__ == "__main__":
    main()
