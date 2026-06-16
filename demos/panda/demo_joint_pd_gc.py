# demos/panda/demo_joint_pd_gc.py

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
    has_affine_position_actuators,
    has_position_actuators_and_neutralize,
)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/panda/panda.xml")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--out-dir", type=str, default="results/dynamics")
    parser.add_argument(
        "--use-ctrl",
        action="store_true",
        help="Use data.ctrl instead of qfrc_applied. Only use this if your XML has torque motor actuators.",
    )
    return parser


def reset_robot(model, data, q0):
    data.qpos[:7] = q0
    data.qvel[:7] = 0.0
    data.qacc[:7] = 0.0
    if model.nu > 0:
        data.ctrl[:] = 0.0
    data.qfrc_applied[:] = 0.0
    mujoco.mj_forward(model, data)


def run_trial(model_path, mode_name, gravity_comp, duration, use_ctrl):
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    if model.nv < 7:
        raise RuntimeError(f"Expected at least 7 DoF, but model.nv={model.nv}")

    q0 = np.array([0.0, -0.7, 0.0, -2.2, 0.0, 1.6, 0.8])
    q_des = q0 + np.array([0.20, -0.15, 0.15, -0.10, 0.10, -0.10, 0.08])

    reset_robot(model, data, q0)

    torque_limit = TorqueLimit(
        lower=-np.array([87, 87, 87, 87, 12, 12, 12], dtype=float),
        upper=np.array([87, 87, 87, 87, 12, 12, 12], dtype=float),
    )

    controller = PandaTorqueController(
        model=model,
        data=data,
        dof=7,
        body_name="hand",
        torque_limit=torque_limit,
    )

    kp = np.array([80, 80, 70, 60, 30, 25, 20], dtype=float)
    kd = np.array([14, 14, 12, 10, 6, 5, 4], dtype=float)

    dt = model.opt.timestep
    steps = int(duration / dt)

    has_position_actuator = not use_ctrl and has_affine_position_actuators(model, 7)
    if has_position_actuator:
        print(
            f"[{mode_name}] Detected affine-bias position actuators in XML. "
            "Neutralizing via ctrl = qpos (qfrc_applied mode)."
        )
    else:
        ctrl_mode = "data.ctrl" if use_ctrl else "qfrc_applied"
        print(f"[{mode_name}] Using {ctrl_mode} for torque application.")

    rows = []

    for k in range(steps):
        t = k * dt

        tau = controller.joint_pd(
            q_des=q_des,
            kp=kp,
            kd=kd,
            gravity_comp=gravity_comp,
        )

        controller.apply_torque(tau, prefer_ctrl=use_ctrl)
        if has_position_actuator:
            has_position_actuators_and_neutralize(model, data, 7)
        mujoco.mj_step(model, data)

        q = data.qpos[:7].copy()
        qd = data.qvel[:7].copy()
        err = q_des - q

        rows.append(
            {
                "t": t,
                "mode": mode_name,
                "err_norm": float(np.linalg.norm(err)),
                "tau_norm": float(np.linalg.norm(tau)),
                **{f"q{i+1}": float(q[i]) for i in range(7)},
                **{f"qd{i+1}": float(qd[i]) for i in range(7)},
                **{f"tau{i+1}": float(tau[i]) for i in range(7)},
            }
        )

    return rows


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = make_parser().parse_args()

    rows_pd = run_trial(
        model_path=args.model,
        mode_name="pd_only",
        gravity_comp=False,
        duration=args.duration,
        use_ctrl=args.use_ctrl,
    )

    rows_gc = run_trial(
        model_path=args.model,
        mode_name="pd_gc",
        gravity_comp=True,
        duration=args.duration,
        use_ctrl=args.use_ctrl,
    )

    out_dir = Path(args.out_dir)
    write_csv(rows_pd, out_dir / "joint_pd_only.csv")
    write_csv(rows_gc, out_dir / "joint_pd_gc.csv")

    print("Saved:")
    print(out_dir / "joint_pd_only.csv")
    print(out_dir / "joint_pd_gc.csv")
    print()
    print("Next: python scripts/plot_dynamics_results.py")


if __name__ == "__main__":
    main()
