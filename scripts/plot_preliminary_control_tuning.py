# scripts/plot_preliminary_control_tuning.py
"""Plot preliminary control tuning results."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_metrics(path):
    """Load metrics CSV and return list of dicts."""
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8", deletechars="")
    rows = []
    for r in data:
        row = {}
        for name in r.dtype.names:
            val = r[name]
            try:
                row[name] = float(val)
            except (ValueError, TypeError):
                row[name] = str(val)
        rows.append(row)
    return rows


def load_raw(path):
    """Load raw CSV and return dict of arrays."""
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8", invalid_raise=False)
    cols = {}
    if data is None or len(data) == 0:
        return cols
    for name in data.dtype.names:
        col = data[name]
        if col.dtype.kind in ("U", "S", "O"):
            numeric = np.full(len(col), np.nan, dtype=np.float64)
            for i, v in enumerate(col):
                try:
                    numeric[i] = float(v)
                except (ValueError, TypeError):
                    pass
            cols[name] = numeric
        else:
            cols[name] = col.astype(np.float64)
    return cols


def plot_error_trajectory(all_raw, out_dir):
    """EE position error over time for each mode."""
    fig, ax = plt.subplots(figsize=(10, 5))
    has_data = False
    for mode, cols in sorted(all_raw.items()):
        t = cols.get("timestamp", np.arange(len(cols.get("ee_position_error", []))))
        err = cols.get("ee_position_error", [])
        if len(err) == 0:
            continue
        has_data = True
        ax.plot(t, err, label=mode, linewidth=1.0, alpha=0.85)
    if not has_data:
        print("  [skip] no ee_position_error data")
        plt.close(fig)
        return
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("EE position error [m]")
    ax.set_title("End-Effector Position Error by Control Mode")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = out_dir / "preliminary_control_error.png"
    fig.savefig(p, dpi=150)
    print("  Saved:", p)
    plt.close(fig)


def plot_torque_trajectory(all_raw, out_dir):
    """Torque norm over time for each mode."""
    fig, ax = plt.subplots(figsize=(10, 5))
    has_data = False
    for mode, cols in sorted(all_raw.items()):
        t = cols.get("timestamp", np.arange(len(cols.get("torque_norm", []))))
        tau = cols.get("torque_norm", [])
        if len(tau) == 0:
            continue
        has_data = True
        ax.plot(t, tau, label=mode, linewidth=1.0, alpha=0.85)
    if not has_data:
        print("  [skip] no torque_norm data")
        plt.close(fig)
        return
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Torque norm [Nm]")
    ax.set_title("Applied Torque Norm by Control Mode")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = out_dir / "preliminary_control_torque.png"
    fig.savefig(p, dpi=150)
    print("  Saved:", p)
    plt.close(fig)


def plot_summary(metrics_rows, out_dir):
    """Bar chart comparing key metrics across modes."""
    modes = [r["mode"] for r in metrics_rows]
    keys = [("mean_ee_position_error", "Mean EE error [m]"),
            ("max_ee_position_error", "Max EE error [m]"),
            ("mean_torque_norm", "Mean torque [Nm]"),
            ("torque_smoothness", "Torque smoothness")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5))
    for ax_idx, (key, ylabel) in enumerate(keys):
        ax = axes[ax_idx]
        vals = [r.get(key, 0) for r in metrics_rows]
        bars = ax.bar(modes, vals, color="steelblue", edgecolor="white", width=0.6)
        ax.set_xticklabels(modes, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(key.replace("_", " ").title(), fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.4f}", ha="center", va="bottom", fontsize=6.5, rotation=45)
    fig.tight_layout()
    p = out_dir / "preliminary_control_summary.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print("  Saved:", p)
    plt.close(fig)


def plot_score(metrics_rows, out_dir):
    """Bar chart of scores sorted ascending."""
    sorted_rows = sorted(metrics_rows, key=lambda r: r.get("score", float("inf")))
    modes = [r["mode"] for r in sorted_rows]
    scores = [r.get("score", 0) for r in sorted_rows]
    diverged = [r.get("diverged", 0) for r in sorted_rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(modes, scores, color="steelblue", edgecolor="white", width=0.6)
    ax.set_xlabel("Mode (sorted by score)")
    ax.set_ylabel("Score (lower is better)")
    ax.set_title("Preliminary Control: Composite Score")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, val, div in zip(bars, scores, diverged):
        label = f"{val:.2f}" + (" (diverged)" if div else "")
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                label, ha="center", va="bottom", fontsize=8, rotation=45)
    fig.tight_layout()
    p = out_dir / "preliminary_control_score.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print("  Saved:", p)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot preliminary control tuning results")
    parser.add_argument("--data-dir", type=str, default="results/preliminary_control")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = Path(__file__).resolve().parents[1] / data_dir
    raw_dir = data_dir / "raw"
    figures_dir = data_dir / "figures"
    metrics_dir = data_dir / "metrics"
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = metrics_dir / "preliminary_control_metrics.csv"
    if not metrics_path.exists():
        print("Run scripts/run_preliminary_control_tuning.py first.")
        print("  ", metrics_path, "not found")
        return
    metrics_rows = load_metrics(str(metrics_path))
    print("Loaded metrics for", len(metrics_rows), "modes")

    all_raw = {}
    for r in metrics_rows:
        mode = r["mode"]
        raw_path = raw_dir / f"preliminary_control_{mode}.csv"
        if raw_path.exists():
            all_raw[mode] = load_raw(str(raw_path))
            print("  raw:", mode, "(", len(all_raw[mode].get("timestamp", [])), "steps)")
        else:
            print("  [warn]", raw_path, "not found, skipping", mode)

    print()
    print("Generating figures:")
    if all_raw:
        plot_error_trajectory(all_raw, figures_dir)
        plot_torque_trajectory(all_raw, figures_dir)
    plot_summary(metrics_rows, figures_dir)
    plot_score(metrics_rows, figures_dir)

    print()
    print("All plots saved to:", figures_dir)


if __name__ == "__main__":
    main()