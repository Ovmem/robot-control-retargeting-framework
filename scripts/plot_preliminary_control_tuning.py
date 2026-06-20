# scripts/plot_preliminary_control_tuning.py
"""Plot preliminary control tuning results."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = Path("results/preliminary_control")


def main():
    metrics_path = DATA_DIR / "metrics" / "control_ablation_metrics.csv"
    if not metrics_path.exists():
        print(f"Run scripts/run_preliminary_control_tuning.py first.\n  {metrics_path} not found")
        return

    d = np.genfromtxt(str(metrics_path), delimiter=",", names=True, dtype=None, encoding="utf-8", deletechars="")

    modes = [r[0] for r in d]
    mean_err = [float(r[1]) for r in d]
    rms_tau = [float(r[4]) for r in d]
    smooth = [float(r[6]) for r in d]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, vals, title, ylabel in zip(
        axes,
        [mean_err, rms_tau, smooth],
        ["Mean joint error", "RMS torque", "Torque smoothness"],
        ["[rad]", "[Nm]", ""],
    ):
        colors = ["#E24A33", "#348ABD", "#988ED5", "#F5A623"][:len(modes)]
        ax.bar(modes, vals, color=colors, edgecolor="white")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = DATA_DIR / "figures" / "preliminary_control_summary.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
