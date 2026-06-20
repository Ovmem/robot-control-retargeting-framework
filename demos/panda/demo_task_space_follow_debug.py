# demos/panda/demo_task_space_follow_debug.py

"""
Debug demo:
No camera, no MediaPipe, no hand retargeting.

It only checks whether the torque-actuated Panda model can follow
a small deterministic end-effector target trajectory.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from core.dynamics_control import (  # noqa: E402
    PandaTorqueController,
    TorqueLimit,
    get_body_pose,
    has_affine_position_actuators,
    print_actuator_diagnostics,
)


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Debug Panda task-space torque following.")
    p.add_argument("--model", type=str, default="models/panda/panda_torque.xml")
    p.add_argument("--duration", type=float, default=8.0)
    p.add_argument("--sim-substeps", type=int, default=10)
    p.add_argument("--output", type=str, default="results/debug_task_space_follow.csv")

    p.add_argument("--kp-pos", type=float, default=250.0)
    p.add_argument("--kd-pos", type=float, default=35.0)
    p.add_argument("--kp-ori", type=float, default=25.0)
    p.add_argument("--kd-ori", type=float, default=4.0)

    p.add_argument(
        "--torque-limit",
        type=float,
        default=None,
        help="Uniform torque limit. If omitted, use Panda-like per-joint limits.",
    )

    return p


def target_trajectory(base_pos: np.ndarray, t: float) -> np.ndarray:
    """
    Small deterministic end-effector target trajectory.

    Keep this small. It is only for verifying the torque control loop.
    """
    return base_pos + np.array(
        [
            0.00,
            0.05 * np.sin(0.8 * t),
            0.03 * np.sin(0.6 * t),
        ],
        dtype=float,
    )


def main() -> None:
    args = make_parser().parse_args()

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    print("\n=== Task-space Follow Debug ===")
    print(f"Model: {model_path}")
    print("Control mode: torque actuator via data.ctrl[:7]")
    print("===============================\n")

    print_actuator_diagnostics(model, dof=7)

    if has_affine_position_actuators(model, dof=7):
        raise RuntimeError(
            "This debug demo requires a torque-actuated Panda model. "
            "Please use --model models/panda/panda_torque.xml."
        )

    q_home = np.array([0.0, -0.7, 0.0, -2.2, 0.0, 1.6, 0.8], dtype=float)

    data.qpos[:7] = q_home
    data.qvel[:7] = 0.0
    data.ctrl[:] = 0.0
    data.qfrc_applied[:] = 0.0
    mujoco.mj_forward(model, data)

    body_name = "hand"
    base_pos, base_rot = get_body_pose(model, data, body_name)

    if args.torque_limit is None:
        torque_limit = TorqueLimit.panda_default()
    else:
        torque_limit = TorqueLimit.uniform(args.torque_limit, dof=7)

    controller = PandaTorqueController(
        model=model,
        data=data,
        dof=7,
        body_name=body_name,
        torque_limit=torque_limit,
    )

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "timestamp",
        "target_x",
        "target_y",
        "target_z",
        "actual_x",
        "actual_y",
        "actual_z",
        "ee_position_error",
        "torque_norm",
        "max_abs_torque",
    ]

    csv_fh = open(output_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_fh, fieldnames=fields)
    writer.writeheader()

    start = time.time()
    frame_id = 0

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                elapsed = time.time() - start
                if elapsed >= args.duration:
                    break

                target_pos = target_trajectory(base_pos, elapsed)
                target_rot = base_rot.copy()

                last_tau = np.zeros(7)

                for _ in range(args.sim_substeps):
                    tau = controller.task_space_pd(
                        target_pos=target_pos,
                        target_rot=target_rot,
                        kp_pos=args.kp_pos,
                        kd_pos=args.kd_pos,
                        kp_rot=args.kp_ori,
                        kd_rot=args.kd_ori,
                        gravity_comp=True,
                    )
                    last_tau = controller.apply_torque(tau, prefer_ctrl=True)

                    # Keep gripper neutral if present.
                    if model.nu >= 8:
                        data.ctrl[7] = 0.0

                    mujoco.mj_step(model, data)

                actual_pos, _ = get_body_pose(model, data, body_name)
                err = float(np.linalg.norm(target_pos - actual_pos))
                tau_norm = float(np.linalg.norm(last_tau))
                max_abs_tau = float(np.max(np.abs(last_tau)))

                writer.writerow(
                    {
                        "timestamp": f"{elapsed:.6f}",
                        "target_x": f"{target_pos[0]:.6f}",
                        "target_y": f"{target_pos[1]:.6f}",
                        "target_z": f"{target_pos[2]:.6f}",
                        "actual_x": f"{actual_pos[0]:.6f}",
                        "actual_y": f"{actual_pos[1]:.6f}",
                        "actual_z": f"{actual_pos[2]:.6f}",
                        "ee_position_error": f"{err:.6f}",
                        "torque_norm": f"{tau_norm:.6f}",
                        "max_abs_torque": f"{max_abs_tau:.6f}",
                    }
                )

                print(
                    f"frame={frame_id:05d} "
                    f"target=({target_pos[0]:.3f},{target_pos[1]:.3f},{target_pos[2]:.3f}) "
                    f"actual=({actual_pos[0]:.3f},{actual_pos[1]:.3f},{actual_pos[2]:.3f}) "
                    f"err={err:.4f}m "
                    f"tau={tau_norm:.2f}Nm",
                    flush=True,
                )

                viewer.sync()
                frame_id += 1

    finally:
        csv_fh.close()

    print(f"\nSaved debug CSV: {output_path}")


if __name__ == "__main__":
    main()
