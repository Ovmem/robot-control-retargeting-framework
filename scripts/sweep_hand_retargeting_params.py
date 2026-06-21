# scripts/sweep_hand_retargeting_params.py
"""Compare recorded demo runs and analyze how different demo parameters
affect mapping stability and tracking response.

Supports single-CSV analysis and multi-CSV comparison.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REQUIRED_METRICS = [
    "detection_rate", "mean_ee_position_error", "max_ee_position_error",
    "final_ee_position_error", "target_position_smoothness",
    "target_position_jump_count", "gripper_command_smoothness",
    "mean_torque_norm", "max_torque_norm", "workspace_clip_rate", "score",
]


def load_csv(path):
    """Load CSV and return dict of column arrays."""
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None,
                         encoding="utf-8", invalid_raise=False)
    cols = {}
    if data is None or len(data) == 0:
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


def compute_metrics(cols, dt_avg=1.0/30.0):
    """Compute metrics from loaded run data."""
    m = {}
    detected = cols.get("detected_hand", np.array([]))
    n = len(detected) if len(detected) > 0 else 1
    m["num_frames"] = int(n)
    m["duration_sec"] = float(n * dt_avg)
    m["detection_rate"] = float(np.mean(detected)) if len(detected) > 0 else np.nan

    ee_err = cols.get("ee_position_error", np.array([]))
    if len(ee_err) > 0:
        m["mean_ee_position_error"] = float(np.nanmean(ee_err))
        m["max_ee_position_error"] = float(np.nanmax(ee_err))
        m["final_ee_position_error"] = float(ee_err[-1]) if len(ee_err) > 0 else np.nan
    else:
        m["mean_ee_position_error"] = m["max_ee_position_error"] = np.nan
        m["final_ee_position_error"] = np.nan

    sx = cols.get("target_pos_x", np.array([]))
    sy = cols.get("target_pos_y", np.array([]))
    sz = cols.get("target_pos_z", np.array([]))
    if len(sx) > 0:
        pos = np.column_stack([sx, sy, sz])
        diffs = np.diff(pos, axis=0)
        step_norms = np.linalg.norm(diffs, axis=1)
        m["target_position_smoothness"] = float(np.nanmean(step_norms))
        m["target_position_jump_count"] = int(np.sum(step_norms > 0.05))
    else:
        m["target_position_smoothness"] = np.nan
        m["target_position_jump_count"] = np.nan

    clipped = cols.get("workspace_clipped", np.array([]))
    if len(clipped) > 0 and not np.all(np.isnan(clipped)):
        m["workspace_clip_rate"] = float(np.mean(clipped > 0.5))
    else:
        m["workspace_clip_rate"] = np.nan

    gw = cols.get("gripper_width", np.array([]))
    if len(gw) > 0:
        gw_diff = np.abs(np.diff(gw))
        m["gripper_command_smoothness"] = float(np.nanmean(gw_diff))
    else:
        m["gripper_command_smoothness"] = np.nan

    tn = cols.get("torque_norm", np.array([]))
    if len(tn) > 0:
        m["mean_torque_norm"] = float(np.nanmean(tn))
        m["max_torque_norm"] = float(np.nanmax(tn))
    else:
        m["mean_torque_norm"] = m["max_torque_norm"] = np.nan

    return m


def compute_score(m):
    """Heuristic score for comparing runs. Lower is better."""
    score = (
        m.get("mean_ee_position_error", 0.1) / 0.10
        + 0.5 * m.get("max_ee_position_error", 0.2) / 0.20
        + 0.2 * m.get("mean_torque_norm", 50) / 50
        + 0.2 * m.get("target_position_smoothness", 0.03) / 0.03
        + 0.2 * m.get("target_position_jump_count", 0)
    )
    return round(score, 4)


def print_summary(label, m):
    """Print one run metrics summary."""
    print(f"  Label: {label}")
    print(f"    Frames: {m.get("num_frames", "?")}")
    print(f"    Duration: {m.get("duration_sec", "?"):.2f}s")
    print(f"    Detection rate: {m.get("detection_rate", "?")}")
    print(f"    Mean EE error: {m.get("mean_ee_position_error", "?"):.4f}m")
    print(f"    Max EE error: {m.get("max_ee_position_error", "?"):.4f}m")
    print(f"    Final EE error: {m.get("final_ee_position_error", "?"):.4f}m")
    print(f"    Target smoothness: {m.get("target_position_smoothness", "?"):.6f}m")
    print(f"    Jump count: {m.get("target_position_jump_count", "?")}")
    print(f"    Gripper smoothness: {m.get("gripper_command_smoothness", "?"):.6f}")
    print(f"    Mean torque norm: {m.get("mean_torque_norm", "?"):.2f}Nm")
    print(f"    Max torque norm: {m.get("max_torque_norm", "?"):.2f}Nm")
    print(f"    Workspace clip rate: {m.get("workspace_clip_rate", "?")}")
    print(f"    Score (lower better): {m.get("score", "?")}")
    print()


def plot_comparison(all_metrics, labels, out_dir):
    """Generate comparison figures."""
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    keys_labels = [
        ("mean_ee_position_error", "Mean EE Error [m]"),
        ("max_ee_position_error", "Max EE Error [m]"),
        ("target_position_smoothness", "Target Smoothness [m]"),
        ("mean_torque_norm", "Mean Torque Norm [Nm]"),
    ]

    for key, ylabel in keys_labels:
        fig, ax = plt.subplots(figsize=(8, 4))
        vals = [m.get(key, 0) for m in all_metrics]
        ax.bar(labels, vals, color="steelblue", edgecolor="white", width=0.5)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(key.replace("_", " ").title())
        ax.grid(True, axis="y", alpha=0.3)
        fname = f"comparison_{key}.png"
        fig.tight_layout()
        fig.savefig(figures_dir / fname, dpi=150)
        print(f"  Saved: {figures_dir / fname}")
        plt.close(fig)

    # Summary: score bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    scores = [m.get("score", m.get("_score", 0)) for m in all_metrics]
    bars = ax.bar(labels, scores, color="steelblue", edgecolor="white", width=0.5)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Score (lower is better)")
    ax.set_title("Run Comparison: Composite Score")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "comparison_summary.png", dpi=150)
    print(f"  Saved: {figures_dir / 'comparison_summary.png'}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare recorded hand retargeting runs")
    parser.add_argument("--input", type=str, default=None,
                        help="Single CSV path")
    parser.add_argument("--inputs", type=str, nargs="+", default=None,
                        help="Multiple CSV paths with labels: path:label path:label ...")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: results/hand_retargeting/comparison)")
    args = parser.parse_args()

    if args.input is None and args.inputs is None:
        print("ERROR: provide --input or --inputs")
        return

    if args.input is not None:
        csv_paths_labels = [(args.input, "run")]
    else:
        csv_paths_labels = []
        for item in args.inputs:
            parts = item.rsplit(":", 1)
            if len(parts) == 2 and parts[1]:
                csv_paths_labels.append((parts[0], parts[1]))
            else:
                csv_paths_labels.append((item, item.split("/")[-3] if "/" in item else item))

    # Determine output dir
    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
    elif args.input is not None:
        csv_path = Path(args.input)
        out_dir = csv_path.parents[1] / "comparison"
    else:
        out_dir = Path("results/hand_retargeting/comparison")
        if not out_dir.is_absolute():
            out_dir = Path(__file__).resolve().parents[1] / out_dir

    all_metrics = []
    labels = []

    for csv_path_str, label in csv_paths_labels:
        csv_path = Path(csv_path_str)
        if not csv_path.exists():
            print(f"  [skip] {csv_path} not found")
            continue

        print(f"\nLoading: {csv_path}")
        cols = load_csv(str(csv_path))
        if not cols:
            print("  [skip] empty CSV")
            continue

        t = cols.get("timestamp", np.array([]))
        dt_avg = 1.0 / 30.0
        if len(t) > 1:
            dt_avg = float(np.median(np.diff(t)))
        print(f"  {len(t)} frames, dt ~ {dt_avg:.3f}s")

        m = compute_metrics(cols, dt_avg)
        m["score"] = compute_score(m)
        all_metrics.append(m)
        labels.append(label)

        print_summary(label, m)

    if not all_metrics:
        print("No valid data loaded.")
        return

    # Save comparison CSV
    raw_dir = out_dir / "raw"
    metrics_dir = out_dir / "metrics"
    raw_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    comp_path = raw_dir / "run_comparison_metrics.csv"
    with open(comp_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label"] + REQUIRED_METRICS)
        for label, m in zip(labels, all_metrics):
            row = [label] + [m.get(k, "") for k in REQUIRED_METRICS]
            w.writerow(row)
    print(f"Saved: {comp_path}")

    # Find best run
    best_idx = int(np.argmin([m.get("score", float("inf")) for m in all_metrics]))
    best_path = metrics_dir / "best_run.csv"
    with open(best_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        m = all_metrics[best_idx]
        w.writerow(["label", labels[best_idx]])
        for k in REQUIRED_METRICS:
            w.writerow([k, m.get(k, "")])
    print(f"Best run: {labels[best_idx]} (score={all_metrics[best_idx].get("score", ""):.4f})")
    print(f"Saved: {best_path}")

    # Generate figures if multiple runs
    if len(all_metrics) > 1:
        print("\nGenerating comparison figures:")
        plot_comparison(all_metrics, labels, out_dir)

    print(f"\nResults in: {out_dir}")


if __name__ == "__main__":
    main()