# scripts/run_panda_control_ablation.py

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import mujoco
import numpy as np

from core.dynamics_control import (
    PandaTorqueController,
    TorqueLimit,
    has_affine_position_actuators,
    has_position_actuators_and_neutralize,
)

# ---------------------------------------------------------------------------
# Target trajectory
# ---------------------------------------------------------------------------
Q0 = np.array([0.0, -0.7, 0.0, -2.2, 0.0, 1.6, 0.8], dtype=float)
Q_STEP = Q0 + np.array([0.20, -0.15, 0.15, -0.10, 0.10, -0.10, 0.08], dtype=float)

DEFAULT_TORQUE_LIMIT = TorqueLimit(
    lower=-np.array([87, 87, 87, 87, 12, 12, 12], dtype=float),
    upper=np.array([87, 87, 87, 87, 12, 12, 12], dtype=float),
)

STRICT_TORQUE_LIMIT = TorqueLimit(
    lower=-np.array([10, 10, 10, 10, 3, 3, 3], dtype=float),
    upper=np.array([10, 10, 10, 10, 3, 3, 3], dtype=float),
)

DEFAULT_KP = np.array([80, 80, 70, 60, 30, 25, 20], dtype=float)
LOW_KP = np.array([40, 40, 35, 30, 15, 12, 10], dtype=float)
DEFAULT_KD = np.array([14, 14, 12, 10, 6, 5, 4], dtype=float)
LOW_KD = np.array([7, 7, 6, 5, 3, 2.5, 2], dtype=float)


def target_trajectory(t: float) -> np.ndarray:
    """Multi-phase joint-space target trajectory.

    - t < 0.5: hold at Q0 (initial settling)
    - 0.5 <= t < 1.0: smooth cubic transition from Q0 to Q_STEP
    - t >= 1.0: hold Q_STEP with a small sinusoidal modulation
    """
    if t < 0.5:
        return Q0.copy()
    elif t < 1.0:
        s = (t - 0.5) / 0.5  # 0 -> 1
        s2 = s * s
        s3 = s2 * s
        smooth = 3.0 * s2 - 2.0 * s3  # Hermite basis
        return Q0 + smooth * (Q_STEP - Q0)
    else:
        # Small sinusoidal modulation around Q_STEP
        delta = 0.03 * np.sin(2.0 * np.pi * 0.3 * (t - 1.0))
        return Q_STEP + np.full(7, delta)


def reset_robot(model, data):
    data.qpos[:7] = Q0
    data.qvel[:7] = 0.0
    data.qacc[:7] = 0.0
    if model.nu > 0:
        data.ctrl[:] = 0.0
    data.qfrc_applied[:] = 0.0
    mujoco.mj_forward(model, data)


def run_trial(
    model_path: str,
    mode_name: str,
    duration: float,
    **kwargs,
) -> List[Dict]:
    """Run a single control trial.

    Extra kwargs are unpacked into the controller call (e.g. kp, kd,
    gravity_comp, torque_limit).
    """
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    if model.nv < 7:
        raise RuntimeError(f"Expected >=7 DoF, got nv={model.nv}")

    reset_robot(model, data)

    torque_limit = kwargs.get("torque_limit", DEFAULT_TORQUE_LIMIT)
    controller = PandaTorqueController(
        model=model,
        data=data,
        dof=7,
        body_name="hand",
        torque_limit=torque_limit,
    )

    dt = model.opt.timestep
    steps = int(duration / dt)

    has_position_actuator = has_affine_position_actuators(model, 7)
    if has_position_actuator:
        print(f"  [{mode_name}] neutralizing position actuators (qfrc_applied mode)")

    rows = []
    tau_prev = None

    for k in range(steps):
        t = k * dt
        q_des = target_trajectory(t)

        # Determine control mode -------------------------------------------------
        control_type = kwargs.get("control_type", "joint_pd")

        if control_type == "joint_pd":
            kp = kwargs.get("kp", DEFAULT_KP)
            kd = kwargs.get("kd", DEFAULT_KD)
            gravity_comp = kwargs.get("gravity_comp", True)
            tau = controller.joint_pd(
                q_des=q_des, kp=kp, kd=kd, gravity_comp=gravity_comp,
            )

        elif control_type == "computed_torque":
            kp = kwargs.get("kp", DEFAULT_KP)
            kd = kwargs.get("kd", DEFAULT_KD)
            tau = controller.computed_torque(
                q_des=q_des, kp=kp, kd=kd,
            )

        elif control_type == "task_space_pd":
            # For task-space we need a body target; approximate from FK at q_des
            data.qpos[:7] = q_des
            mujoco.mj_forward(model, data)
            body_id = model.body("hand").id
            target_pos = data.xpos[body_id].copy()
            target_rot = data.xmat[body_id].reshape(3, 3).copy()
            # restore actual state
            data.qpos[:7] = controller.q()
            mujoco.mj_forward(model, data)

            tau = controller.task_space_pd(
                target_pos=target_pos,
                target_rot=target_rot,
                kp_pos=kwargs.get("kp_pos", 250.0),
                kd_pos=kwargs.get("kd_pos", 30.0),
                kp_rot=kwargs.get("kp_rot", 40.0),
                kd_rot=kwargs.get("kd_rot", 6.0),
                gravity_comp=kwargs.get("gravity_comp", True),
            )

        else:
            raise ValueError(f"Unknown control_type: {control_type}")

        # Apply and step ---------------------------------------------------------
        controller.apply_torque(tau, prefer_ctrl=False)
        if has_position_actuator:
            has_position_actuators_and_neutralize(model, data, 7)
        mujoco.mj_step(model, data)

        q_cur = data.qpos[:7].copy()
        qd_cur = data.qvel[:7].copy()
        err = q_des - q_cur

        # Torque smoothness: norm of tau difference from previous step
        tau_smoothness_step = 0.0
        if tau_prev is not None:
            tau_smoothness_step = float(np.linalg.norm(tau - tau_prev))
        tau_prev = tau.copy()

        rows.append({
            "t": float(t),
            "mode": mode_name,
            "err_norm": float(np.linalg.norm(err)),
            "tau_norm": float(np.linalg.norm(tau)),
            "tau_smoothness_step": tau_smoothness_step,
        })
        for i in range(7):
            rows[-1][f"q{i+1}"] = float(q_cur[i])
            rows[-1][f"qd{i+1}"] = float(qd_cur[i])
            rows[-1][f"tau{i+1}"] = float(tau[i])

    return rows


def compute_metrics(rows: List[Dict]) -> Dict[str, float]:
    """Compute aggregate metrics from per-step records."""
    err_norms = np.array([r["err_norm"] for r in rows])
    tau_norms = np.array([r["tau_norm"] for r in rows])
    tau_smooth_steps = np.array([r["tau_smoothness_step"] for r in rows])

    metrics = {
        "mean_joint_error": float(np.mean(err_norms)),
        "final_joint_error": float(err_norms[-1]),
        "max_joint_error": float(np.max(err_norms)),
        "rms_torque": float(np.sqrt(np.mean(tau_norms ** 2))),
        "max_torque": float(np.max(tau_norms)),
        "torque_smoothness": float(np.mean(tau_smooth_steps[tau_smooth_steps > 0])),
        "overshoot": float(np.max(err_norms) - err_norms[-1]),
        "diverged": float(1.0 if (np.any(np.isnan(err_norms)) or np.any(np.isinf(err_norms)) or np.any(np.isnan(tau_norms)) or np.max(err_norms) > 1.0 or np.max(tau_norms) > 100.0) else 0.0),
    }
    return metrics


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_metrics_csv(all_metrics: Dict[str, Dict], path):
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered_metrics = [
        "mean_joint_error", "final_joint_error", "max_joint_error",
        "rms_torque", "max_torque", "torque_smoothness",
        "overshoot", "diverged",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["mode"] + ordered_metrics)
        for mode, m in all_metrics.items():
            writer.writerow([mode] + [m.get(k, float("nan")) for k in ordered_metrics])


DEFINITIONS = [
    {
        "mode": "pd_only",
        "control_type": "joint_pd",
        "gravity_comp": False,
        "kp": DEFAULT_KP,
        "kd": DEFAULT_KD,
        "torque_limit": DEFAULT_TORQUE_LIMIT,
    },
    {
        "mode": "pd_gc",
        "control_type": "joint_pd",
        "gravity_comp": True,
        "kp": DEFAULT_KP,
        "kd": DEFAULT_KD,
        "torque_limit": DEFAULT_TORQUE_LIMIT,
    },
    {
        "mode": "pd_gc_low_gain",
        "control_type": "joint_pd",
        "gravity_comp": True,
        "kp": LOW_KP,
        "kd": LOW_KD,
        "torque_limit": DEFAULT_TORQUE_LIMIT,
    },
    {
        "mode": "pd_gc_torque_clipped",
        "control_type": "joint_pd",
        "gravity_comp": True,
        "kp": DEFAULT_KP,
        "kd": DEFAULT_KD,
        "torque_limit": STRICT_TORQUE_LIMIT,
    },
    {
        "mode": "computed_torque",
        "control_type": "computed_torque",
        "kp": DEFAULT_KP,
        "kd": DEFAULT_KD,
        "torque_limit": DEFAULT_TORQUE_LIMIT,
    },
    {
        "mode": "task_space_pd_gc",
        "control_type": "task_space_pd",
        "gravity_comp": True,
        "kp_pos": 250.0,
        "kd_pos": 30.0,
        "kp_rot": 40.0,
        "kd_rot": 6.0,
        "torque_limit": DEFAULT_TORQUE_LIMIT,
    },
]


def main():
    parser = argparse.ArgumentParser(
        description="Run Panda control ablation study")
    parser.add_argument("--model", type=str, default="models/panda/panda.xml")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Simulation duration per trial (seconds)")
    parser.add_argument("--out-dir", type=str,
                        default="results/preliminary_control")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    metrics_dir = out_dir / "metrics"

    print("=" * 60)
    print("Preliminary Control Tuning")
    print(f"  model:    {args.model}")
    print(f"  duration: {args.duration} s per trial")
    print(f"  modes:    {[d['mode'] for d in DEFINITIONS]}")
    print("=" * 60)

    all_metrics = {}

    for cfg in DEFINITIONS:
        mode = cfg["mode"]
        print(f"\n--- Running: {mode} ---")
        rows = run_trial(
            model_path=args.model,
            mode_name=mode,
            duration=args.duration,
            **cfg,
        )

        # Save raw per-step data
        csv_path = raw_dir / f"ablation_{mode}.csv"
        write_csv(rows, csv_path)
        print(f"  saved: {csv_path}")

        # Compute metrics
        metrics = compute_metrics(rows)
        all_metrics[mode] = metrics
        print(f"  mean_joint_error: {metrics['mean_joint_error']:.4f} rad")
        print(f"  final_joint_error: {metrics['final_joint_error']:.4f} rad")
        print(f"  rms_torque: {metrics['rms_torque']:.2f} Nm")
        print(f"  torque_smoothness: {metrics['torque_smoothness']:.4f}")

    # Save aggregate metrics
    metrics_path = metrics_dir / "control_ablation_metrics.csv"
    write_metrics_csv(all_metrics, metrics_path)
    print(f"\nAggregate metrics saved: {metrics_path}")

    print("\nDone. Next: python scripts/plot_preliminary_control_tuning.py")


if __name__ == "__main__":
    main()
