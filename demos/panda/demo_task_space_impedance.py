# demos/panda/demo_task_space_impedance.py

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import csv
from pathlib import Path

import mujoco
import numpy as np

from core.dynamics_control import (
    PandaTorqueController,
    TorqueLimit,
    get_body_pose,
    has_affine_position_actuators,
    has_position_actuators_and_neutralize,
    rotation_error_rotvec,
)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/panda/panda.xml")
    parser.add_argument("--body", type=str, default="hand")
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--out-dir", type=str, default="results/dynamics")
    parser.add_argument("--use-ctrl", action="store_true")
    return parser


def require_body(model, body_name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        names = []
        for i in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name:
                names.append(name)
        raise RuntimeError(
            f"Body '{body_name}' not found.\n"
            f"Available bodies include:\n{names[:50]}"
        )
    return body_id


def reset_robot(model, data, q0):
    data.qpos[:7] = q0
    data.qvel[:7] = 0.0
    data.qacc[:7] = 0.0
    if model.nu > 0:
        data.ctrl[:] = 0.0
    data.qfrc_applied[:] = 0.0
    mujoco.mj_forward(model, data)


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = make_parser().parse_args()

    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)

    require_body(model, args.body)

    q_home = np.array([0.0, -0.7, 0.0, -2.2, 0.0, 1.6, 0.8])
    reset_robot(model, data, q_home)

    torque_limit = TorqueLimit(
        lower=-np.array([87, 87, 87, 87, 12, 12, 12], dtype=float),
        upper=np.array([87, 87, 87, 87, 12, 12, 12], dtype=float),
    )

    controller = PandaTorqueController(
        model=model,
        data=data,
        dof=7,
        body_name=args.body,
        torque_limit=torque_limit,
    )

    base_pos, base_rot = get_body_pose(model, data, args.body)

    dt = model.opt.timestep
    steps = int(args.duration / dt)

    has_position_actuator = not args.use_ctrl and has_affine_position_actuators(model, 7)
    if has_position_actuator:
        print(
            "[task_impedance] Detected affine-bias position actuators in XML. "
            "Neutralizing via ctrl = qpos (qfrc_applied mode)."
        )
    else:
        ctrl_mode = "data.ctrl" if args.use_ctrl else "qfrc_applied"
        print(f"[task_impedance] Using {ctrl_mode} for torque application.")

    rows = []

    for k in range(steps):
        t = k * dt

        # 一个小幅度圆形末端轨迹，幅度不要太大，先保证稳定
        radius_y = 0.04
        radius_z = 0.03
        freq = 0.20

        target_pos = base_pos + np.array(
            [
                0.04 * np.sin(2 * np.pi * freq * t),
                radius_y * np.cos(2 * np.pi * freq * t) - radius_y,
                radius_z * np.sin(2 * np.pi * freq * t),
            ]
        )

        target_rot = base_rot.copy()

        tau = controller.task_space_pd(
            target_pos=target_pos,
            target_rot=target_rot,
            kp_pos=180.0,
            kd_pos=25.0,
            kp_rot=25.0,
            kd_rot=4.0,
            q_null_des=q_home,
            kp_null=4.0,
            kd_null=0.8,
            gravity_comp=True,
        )

        controller.apply_torque(tau, prefer_ctrl=args.use_ctrl)
        if has_position_actuator:
            has_position_actuators_and_neutralize(model, data, 7)
        mujoco.mj_step(model, data)

        cur_pos, cur_rot = get_body_pose(model, data, args.body)

        pos_err = target_pos - cur_pos
        rot_err = rotation_error_rotvec(target_rot, cur_rot)

        rows.append(
            {
                "t": float(t),
                "pos_err_norm": float(np.linalg.norm(pos_err)),
                "rot_err_norm": float(np.linalg.norm(rot_err)),
                "tau_norm": float(np.linalg.norm(tau)),
                "target_x": float(target_pos[0]),
                "target_y": float(target_pos[1]),
                "target_z": float(target_pos[2]),
                "actual_x": float(cur_pos[0]),
                "actual_y": float(cur_pos[1]),
                "actual_z": float(cur_pos[2]),
            }
        )

    out_path = Path(args.out_dir) / "task_space_impedance.csv"
    write_csv(rows, out_path)

    print("Saved:")
    print(out_path)
    print()
    print("Next: python scripts/plot_dynamics_results.py")


if __name__ == "__main__":
    main()
