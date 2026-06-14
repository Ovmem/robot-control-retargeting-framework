# scripts/plot_dynamics_results.py

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")


def plot_joint_error(pd_only, pd_gc, out_dir):
    plt.figure()
    plt.plot(pd_only["t"], pd_only["err_norm"], label="PD only")
    plt.plot(pd_gc["t"], pd_gc["err_norm"], label="PD + gravity compensation")
    plt.xlabel("Time [s]")
    plt.ylabel("Joint error norm [rad]")
    plt.title("Joint-space tracking error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "pd_gc_tracking_curve.png", dpi=200)
    plt.close()


def plot_torque(pd_only, pd_gc, out_dir):
    plt.figure()
    plt.plot(pd_only["t"], pd_only["tau_norm"], label="PD only")
    plt.plot(pd_gc["t"], pd_gc["tau_norm"], label="PD + gravity compensation")
    plt.xlabel("Time [s]")
    plt.ylabel("Torque norm [Nm]")
    plt.title("Joint torque norm")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "pd_gc_torque_curve.png", dpi=200)
    plt.close()


def plot_task_space(task, out_dir):
    plt.figure()
    plt.plot(task["t"], task["pos_err_norm"], label="Position error")
    plt.plot(task["t"], task["rot_err_norm"], label="Orientation error")
    plt.xlabel("Time [s]")
    plt.ylabel("Error norm")
    plt.title("Task-space impedance tracking error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "task_space_impedance_error_curve.png", dpi=200)
    plt.close()


def write_metrics(pd_only, pd_gc, task, out_dir):
    path = out_dir / "metrics.csv"

    rows = [
        [
            "joint_pd_only_mean_error",
            float(np.mean(pd_only["err_norm"])),
        ],
        [
            "joint_pd_gc_mean_error",
            float(np.mean(pd_gc["err_norm"])),
        ],
        [
            "joint_pd_only_final_error",
            float(pd_only["err_norm"][-1]),
        ],
        [
            "joint_pd_gc_final_error",
            float(pd_gc["err_norm"][-1]),
        ],
        [
            "task_space_mean_pos_error",
            float(np.mean(task["pos_err_norm"])),
        ],
        [
            "task_space_final_pos_error",
            float(task["pos_err_norm"][-1]),
        ],
        [
            "task_space_mean_rot_error",
            float(np.mean(task["rot_err_norm"])),
        ],
    ]

    with path.open("w", encoding="utf-8") as f:
        f.write("metric,value\n")
        for key, value in rows:
            f.write(f"{key},{value}\n")

    print("Saved:", path)


def main():
    out_dir = Path("results/dynamics")
    out_dir.mkdir(parents=True, exist_ok=True)

    pd_only_path = out_dir / "joint_pd_only.csv"
    pd_gc_path = out_dir / "joint_pd_gc.csv"
    task_path = out_dir / "task_space_impedance.csv"

    missing = [p for p in [pd_only_path, pd_gc_path, task_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing CSV files:\n"
            + "\n".join(str(p) for p in missing)
            + "\n\nRun these first:\n"
            + "PYTHONPATH=. python demos/panda/demo_joint_pd_gc.py\n"
            + "PYTHONPATH=. python demos/panda/demo_task_space_impedance.py"
        )

    pd_only = read_csv(pd_only_path)
    pd_gc = read_csv(pd_gc_path)
    task = read_csv(task_path)

    plot_joint_error(pd_only, pd_gc, out_dir)
    plot_torque(pd_only, pd_gc, out_dir)
    plot_task_space(task, out_dir)
    write_metrics(pd_only, pd_gc, task, out_dir)

    print("Saved figures:")
    print(out_dir / "pd_gc_tracking_curve.png")
    print(out_dir / "pd_gc_torque_curve.png")
    print(out_dir / "task_space_impedance_error_curve.png")


if __name__ == "__main__":
    main()
