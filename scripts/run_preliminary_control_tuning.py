# scripts/run_preliminary_control_tuning.py
"""Tune task-space control parameters using a fixed end-effector target trajectory.

The goal is to find stable default control parameters for the real hand retargeting demo,
by testing different gains, torque limits, and step-size targets on a known trajectory.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from core.dynamics_control import (
    PandaTorqueController,
    TorqueLimit,
    get_body_pose,
    rotation_error_rotvec,
    has_affine_position_actuators,
)

# ---------------------------------------------------------------------------
# Control mode definitions
# ---------------------------------------------------------------------------
@dataclass
class ControlConfig:
    mode: str
    kp_pos: float
    kd_pos: float
    kp_ori: float
    kd_ori: float
    torque_limit_val: float
    max_target_step: float


CONFIGS: List[ControlConfig] = [
    ControlConfig("soft",          80,  16, 20, 4, 40, 0.025),
    ControlConfig("balanced",     120,  24, 35, 7, 55, 0.035),
    ControlConfig("responsive",   160,  32, 45, 9, 70, 0.045),
    ControlConfig("aggressive",   220,  40, 60, 12, 90, 0.060),
    ControlConfig("torque_limited", 140, 28, 40, 8, 35, 0.035),
]

# ---------------------------------------------------------------------------
# Target trajectory
# ---------------------------------------------------------------------------
Q_HOME = np.array([0.0, -0.7, 0.0, -2.2, 0.0, 1.6, 0.8], dtype=float)


def make_target_pos(initial: np.ndarray, t: float) -> np.ndarray:
    """Return end-effector target position at time t."""
    return initial + np.array([
        0.02 * np.sin(0.5 * 2.0 * np.pi * t),
        0.06 * np.sin(0.8 * 2.0 * np.pi * t),
        0.04 * np.sin(0.6 * 2.0 * np.pi * t),
    ])


RAW_FIELDS = [
    "timestamp", "mode",
    "kp_pos", "kd_pos", "kp_ori", "kd_ori", "torque_limit", "max_target_step",
    "target_ee_pos_x", "target_ee_pos_y", "target_ee_pos_z",
    "actual_ee_pos_x", "actual_ee_pos_y", "actual_ee_pos_z",
    "ee_position_error",
    "target_quat_w", "target_quat_x", "target_quat_y", "target_quat_z",
    "actual_quat_w", "actual_quat_x", "actual_quat_y", "actual_quat_z",
    "ee_orientation_error",
    "torque_norm", "max_abs_torque",
]

METRICS_FIELDS = [
    "mode", "kp_pos", "kd_pos", "kp_ori", "kd_ori",
    "torque_limit", "max_target_step",
    "mean_ee_position_error", "final_ee_position_error", "max_ee_position_error",
    "mean_torque_norm", "max_torque_norm", "torque_smoothness",
    "diverged", "score",
]

BEST_FIELDS = [
    "mode", "kp_pos", "kd_pos", "kp_ori", "kd_ori",
    "torque_limit", "max_target_step",
    "score", "mean_ee_position_error", "max_ee_position_error",
    "mean_torque_norm", "max_torque_norm", "torque_smoothness",
]


def reset_robot(model, data):
    data.qpos[:7] = Q_HOME
    data.qvel[:7] = 0.0
    data.qacc[:7] = 0.0
    if model.nu > 0:
        data.ctrl[:] = 0.0
    data.qfrc_applied[:] = 0.0
    mujoco.mj_forward(model, data)


def run_trial(config: ControlConfig, model_path: str, duration: float) -> List[Dict]:
    """Run a single control trial with the given config and return per-step rows."""
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    reset_robot(model, data)

    torque_limit = TorqueLimit(
        lower=-np.full(7, config.torque_limit_val, dtype=float),
        upper=np.full(7, config.torque_limit_val, dtype=float),
    )
    controller = PandaTorqueController(
        model=model, data=data, dof=7, body_name="hand",
        torque_limit=torque_limit,
    )

    # Get initial end-effector pose
    initial_pos, initial_rot = get_body_pose(model, data, "hand")
    target_rot = initial_rot.copy()

    dt = model.opt.timestep
    steps = int(duration / dt)
    prev_tau = None

    rows = []
    tau_prev = None

    for k in range(steps):
        t = k * dt
        target_pos = make_target_pos(initial_pos, t)

        tau = controller.task_space_pd(
            target_pos=target_pos,
            target_rot=target_rot,
            kp_pos=config.kp_pos,
            kd_pos=config.kd_pos,
            kp_rot=config.kp_ori,
            kd_rot=config.kd_ori,
            gravity_comp=True,
        )

        controller.apply_torque(tau, prefer_ctrl=True)
        mujoco.mj_step(model, data)

        actual_pos, actual_rot = get_body_pose(model, data, "hand")
        pos_err = float(np.linalg.norm(target_pos - actual_pos))
        rot_err = float(np.linalg.norm(rotation_error_rotvec(target_rot, actual_rot)))
        tau_norm = float(np.linalg.norm(tau))
        max_tau = float(np.max(np.abs(tau)))

        target_quat = Rotation.from_matrix(target_rot).as_quat()
        actual_quat = Rotation.from_matrix(actual_rot).as_quat()

        # Smoothness: norm of torque difference from previous step
        ts_step = 0.0
        if tau_prev is not None:
            ts_step = float(np.linalg.norm(tau - tau_prev))
        tau_prev = tau.copy()

        rows.append({
            "timestamp": f"{t:.4f}",
            "mode": config.mode,
            "kp_pos": config.kp_pos,
            "kd_pos": config.kd_pos,
            "kp_ori": config.kp_ori,
            "kd_ori": config.kd_ori,
            "torque_limit": config.torque_limit_val,
            "max_target_step": config.max_target_step,
            "target_ee_pos_x": f"{target_pos[0]:.4f}",
            "target_ee_pos_y": f"{target_pos[1]:.4f}",
            "target_ee_pos_z": f"{target_pos[2]:.4f}",
            "actual_ee_pos_x": f"{actual_pos[0]:.4f}",
            "actual_ee_pos_y": f"{actual_pos[1]:.4f}",
            "actual_ee_pos_z": f"{actual_pos[2]:.4f}",
            "ee_position_error": f"{pos_err:.6f}",
            "target_quat_w": f"{target_quat[3]:.6f}",
            "target_quat_x": f"{target_quat[0]:.6f}",
            "target_quat_y": f"{target_quat[1]:.6f}",
            "target_quat_z": f"{target_quat[2]:.6f}",
            "actual_quat_w": f"{actual_quat[3]:.6f}",
            "actual_quat_x": f"{actual_quat[0]:.6f}",
            "actual_quat_y": f"{actual_quat[1]:.6f}",
            "actual_quat_z": f"{actual_quat[2]:.6f}",
            "ee_orientation_error": f"{rot_err:.6f}",
            "torque_norm": f"{tau_norm:.4f}",
            "max_abs_torque": f"{max_tau:.4f}",
        })

    return rows


def compute_metrics(rows: List[Dict]) -> Dict:
    """Compute aggregate metrics from per-step records."""
    errs = np.array([float(r["ee_position_error"]) for r in rows])
    taus = np.array([float(r["torque_norm"]) for r in rows])

    # Torque smoothness: mean of adjacent-step torque norm diffs
    tau_diff = np.diff(taus)
    smoothness = float(np.mean(np.abs(tau_diff))) if len(tau_diff) > 0 else 0.0

    diverged = bool(
        np.any(np.isnan(errs)) or np.any(np.isinf(errs)) or
        np.any(np.isnan(taus)) or np.any(np.isinf(taus)) or
        np.max(errs) > 0.5
    )

    metrics = {
        "mean_ee_position_error": float(np.mean(errs)),
        "final_ee_position_error": float(errs[-1]),
        "max_ee_position_error": float(np.max(errs)),
        "mean_torque_norm": float(np.mean(taus)),
        "max_torque_norm": float(np.max(taus)),
        "torque_smoothness": smoothness,
        "diverged": diverged,
    }

    # Score: lower is better
    norm_err = metrics["mean_ee_position_error"] / 0.1  # normalize to ~1 for 10cm error
    norm_max = metrics["max_ee_position_error"] / 0.2
    norm_tau = metrics["mean_torque_norm"] / 50.0
    norm_smooth = metrics["torque_smoothness"] / 10.0
    div_penalty = 100.0 if diverged else 0.0

    score = (1.0 * norm_err + 0.5 * norm_max + 0.2 * norm_tau + 0.2 * norm_smooth + div_penalty)
    metrics["score"] = score

    return metrics


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_metrics_csv(all_metrics: Dict[str, Dict], path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(METRICS_FIELDS)
        for cfg in CONFIGS:
            m = all_metrics.get(cfg.mode, {})
            w.writerow([cfg.mode, cfg.kp_pos, cfg.kd_pos, cfg.kp_ori, cfg.kd_ori,
                       cfg.torque_limit_val, cfg.max_target_step] +
                      [m.get(k, "") for k in METRICS_FIELDS[7:]])


def write_best_csv(all_metrics: Dict[str, Dict], path):
    """Write best parameters (lowest score among non-diverged) to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)

    valid = [(cfg.mode, all_metrics.get(cfg.mode, {})) for cfg in CONFIGS
             if not all_metrics.get(cfg.mode, {}).get("diverged", True)]

    if not valid:
        print("WARNING: all modes diverged, no best params found.")
        return

    best_mode, best_m = min(valid, key=lambda x: x[1].get("score", float("inf")))

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=BEST_FIELDS)
        w.writeheader()
        w.writerow({
            "mode": best_mode,
            "kp_pos": next(c.kp_pos for c in CONFIGS if c.mode == best_mode),
            "kd_pos": next(c.kd_pos for c in CONFIGS if c.mode == best_mode),
            "kp_ori": next(c.kp_ori for c in CONFIGS if c.mode == best_mode),
            "kd_ori": next(c.kd_ori for c in CONFIGS if c.mode == best_mode),
            "torque_limit": next(c.torque_limit_val for c in CONFIGS if c.mode == best_mode),
            "max_target_step": next(c.max_target_step for c in CONFIGS if c.mode == best_mode),
            "score": best_m.get("score", ""),
            "mean_ee_position_error": best_m.get("mean_ee_position_error", ""),
            "max_ee_position_error": best_m.get("max_ee_position_error", ""),
            "mean_torque_norm": best_m.get("mean_torque_norm", ""),
            "max_torque_norm": best_m.get("max_torque_norm", ""),
            "torque_smoothness": best_m.get("torque_smoothness", ""),
        })

    print(f"\nBest preliminary control parameters:")
    bc = next(c for c in CONFIGS if c.mode == best_mode)
    print(f"  mode = {best_mode}")
    print(f"  kp_pos = {bc.kp_pos}")
    print(f"  kd_pos = {bc.kd_pos}")
    print(f"  kp_ori = {bc.kp_ori}")
    print(f"  kd_ori = {bc.kd_ori}")
    print(f"  torque_limit = {bc.torque_limit_val}")
    print(f"  max_target_step = {bc.max_target_step}")
    print(f"  score = {best_m.get('score', 'N/A'):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Preliminary control tuning - task-space parameters for hand retargeting")
    parser.add_argument("--model", type=str, default="models/panda/panda.xml")
    parser.add_argument("--duration", type=float, default=8.0,
                        help="Trajectory duration in seconds")
    parser.add_argument("--out-dir", type=str, default="results/preliminary_control")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    metrics_dir = out_dir / "metrics"

    print(f"\nPreliminary Control Tuning")
    print(f"  model:    {args.model}")
    print(f"  duration: {args.duration}s per mode")
    print(f"  modes:    {[c.mode for c in CONFIGS]}")
    print("=" * 50)

    all_metrics = {}

    for cfg in CONFIGS:
        print(f"\n  Running: {cfg.mode}  "
              f"kp_pos={cfg.kp_pos} kd_pos={cfg.kd_pos} "
              f"kp_ori={cfg.kp_ori} kd_ori={cfg.kd_ori} "
              f"limit={cfg.torque_limit_val}")
        rows = run_trial(cfg, args.model, args.duration)

        raw_path = raw_dir / f"preliminary_control_{cfg.mode}.csv"
        write_csv(rows, raw_path)
        print(f"    raw: {raw_path}")

        metrics = compute_metrics(rows)
        all_metrics[cfg.mode] = metrics
        print(f"    mean_err={metrics['mean_ee_position_error']:.4f}m  "
              f"mean_tau={metrics['mean_torque_norm']:.1f}Nm  "
              f"smooth={metrics['torque_smoothness']:.4f}  "
              f"score={metrics['score']:.4f}  "
              f"diverged={metrics['diverged']}")

    # Save metrics CSV
    metrics_path = metrics_dir / "preliminary_control_metrics.csv"
    write_metrics_csv(all_metrics, metrics_path)
    print(f"\n  Metrics: {metrics_path}")

    # Save best params
    best_path = metrics_dir / "best_control_params.csv"
    write_best_csv(all_metrics, best_path)
    print(f"  Best:    {best_path}")

    print("\n  Next: python scripts/plot_preliminary_control_tuning.py")


if __name__ == "__main__":
    main()
