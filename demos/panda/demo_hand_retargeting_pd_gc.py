# demos/panda/demo_hand_retargeting_pd_gc.py

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import time
import argparse
import csv

import mujoco
import mujoco.viewer
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

from core.dynamics_control import (
    PandaTorqueController,
    TorqueLimit,
    get_body_pose,
)
from retargeting.hand_to_panda import HandToPandaRetargeter
from vision.hand_tracker import MediaPipeHandTracker



def make_parser():
    parser = argparse.ArgumentParser(
        description="Webcam hand retargeting demo")
    parser.add_argument("--pos-scale", type=float, default=2.2,
                        help="Hand motion to robot position scale")
    parser.add_argument("--filter-alpha", type=float, default=0.18,
                        help="Low-pass filter strength (0=no smoothing, 1=max)")
    parser.add_argument("--enable-depth-mapping", action="store_true",
                        help="Enable relative depth -> x mapping (experimental)")
    parser.add_argument("--no-debug", action="store_true",
                        help="Disable debug overlay on separate info window")
    parser.add_argument("--log-csv", type=str, default=None,
                        help="Save run log to CSV file path")
    parser.add_argument("--duration", type=float, default=0,
                        help="Auto-stop after N seconds (0 = manual stop)")
    return parser


def main():
    args = make_parser().parse_args()

    model = mujoco.MjModel.from_xml_path("models/panda/panda.xml")
    data = mujoco.MjData(model)

    # A safe initial configuration for Panda.
    q_home = np.array([0.0, -0.7, 0.0, -2.2, 0.0, 1.6, 0.8])
    data.qpos[:7] = q_home
    data.qvel[:7] = 0.0
    if model.nu >= 7:
        data.ctrl[:7] = q_home
    mujoco.mj_forward(model, data)

    body_name = "hand"

    base_pos, base_rot = get_body_pose(model, data, body_name)

    target_pos = base_pos.copy()
    target_rot = base_rot.copy()

    torque_limit = TorqueLimit(
        lower=-np.array([87, 87, 87, 87, 12, 12, 12], dtype=float),
        upper=np.array([87, 87, 87, 87, 12, 12, 12], dtype=float),
    )

    controller = PandaTorqueController(
        model=model,
        data=data,
        dof=7,
        body_name=body_name,
        torque_limit=torque_limit,
    )

    tracker = MediaPipeHandTracker(
        camera_id=0,
        draw=True,
        mirror=True,
    )

    # --- startup info ---
    print("=== Hand Retargeting Demo ===")
    print(f"  pos_scale:       {args.pos_scale}")
    print(f"  filter_alpha:    {args.filter_alpha}")
    print(f"  depth_mapping:   {'ON' if args.enable_depth_mapping else 'OFF'}")
    print(f"  log_csv:         {args.log_csv}")
    print(f"  duration:        {args.duration}s (0 = manual stop)")
    print(f"  workspace clamp: y +/-0.25, z -0.12/+0.14, x +/-0.02")
    print(f"  gripper range:   [0.00, 0.04] m")
    if args.log_csv:
        print(f"  Logging to:      {args.log_csv}")
    print("==============================")

    retargeter = HandToPandaRetargeter(
        robot_origin=base_pos.copy(),
        position_scale_xy=args.pos_scale,
        depth_scale=0.8,
        enable_depth_mapping=args.enable_depth_mapping,
        filter_alpha=args.filter_alpha,
    )

    # Default target before hand is detected.
    target_pos = np.array([0.45, 0.0, 0.45])
    target_gripper = 0.04  # default: fully open
    # target_rot = Rotation.from_euler("xyz", [180, 0, 90], degrees=True).as_matrix()

    print("Press ESC in the MediaPipe window or close MuJoCo viewer to exit.")
    print("Move your wrist to command end-effector position.")
    print("Pinch thumb-index to command gripper width (set via data.ctrl[7]).")

    last_print = time.time()

    sim_substeps = 60  # 可以试 10、15、20

    demo_start = time.time()
    csv_writer = None
    if args.log_csv:
        csv_path = Path(args.log_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fh_csv = open(csv_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(fh_csv, fieldnames=["t", "valid", "target_x", "target_y", "target_z", "gripper_width", "pinch_ratio"])
        csv_writer.writeheader()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():

            obs = tracker.read()

            if obs is not None:
                target = retargeter.update(obs)
                print(
                    "target_pos:",
                    target.pos.round(3),
                    "base_pos:",
                    base_pos.round(3),
                )

                if target.valid:
                    target_pos = target.pos
                    target_rot = target.rot
                    target_gripper = target.gripper_width

            # --- duration auto-stop ---
            if args.duration > 0 and (time.time() - demo_start) > args.duration:
                print(f"Duration limit reached ({args.duration}s), exiting.")
                break

            # --- CSV logging ---
            if csv_writer is not None:
                csv_writer.writerow({
                    "t": f"{time.time() - demo_start:.3f}",
                    "valid": int(target.valid) if "target" in dir() else 0,
                    "target_x": f"{target_pos[0]:.4f}",
                    "target_y": f"{target_pos[1]:.4f}",
                    "target_z": f"{target_pos[2]:.4f}",
                    "gripper_width": f"{target_gripper:.4f}",
                    "pinch_ratio": f"{target.pinch_ratio:.3f}" if hasattr(target, "pinch_ratio") else "",
                })

            # --- debug overlay ---
            if not args.no_debug and obs is not None:
                frame = obs.frame_bgr.copy()
                y = 30
                lines = [
                    f"Valid: {target.valid if 'target' in dir() else False}",
                    f"Target: ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})",
                    f"Gripper: {target_gripper:.4f}",
                    f"Pinch ratio: {target.pinch_ratio:.3f}" if hasattr(target, "pinch_ratio") else "",
                    f"Depth: {'ON' if args.enable_depth_mapping else 'OFF'}",
                    "Press ESC in MediaPipe window to quit",
                ]
                for text in lines:
                    cv2.putText(frame, text, (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                    y += 28
                cv2.imshow("Retargeting Debug", frame)

            for _ in range(sim_substeps):
                    # 如果 XML 里是 position actuator，让它保持当前关节角，避免它把机械臂拉回 ctrl=0
                if model.nu >= 7:
                    data.ctrl[:7] = data.qpos[:7]
                # Gripper control from retargeter
                if model.nu >= 8:
                    g_ctrl = np.clip(target_gripper / 0.04 * 255.0, 0, 255)
                    data.ctrl[7] = g_ctrl

                tau = controller.task_space_pd(
                    target_pos=target_pos,
                    target_rot=target_rot,
                    kp_pos=np.array([1400.0, 1100.0, 1600.0]),
                    kd_pos=np.array([100.0, 85.0, 110.0]),
                    kp_rot=8.0,
                    kd_rot=2.0,
                    q_null_des=None,
                    kp_null=0.0,
                    kd_null=0.0,
                    gravity_comp=True,
                )

                controller.apply_torque(tau, prefer_ctrl=False)
                mujoco.mj_step(model, data)
            
            actual_pos, _ = get_body_pose(model, data, body_name)
            print(
                "target:", target_pos.round(3),
                "actual:", actual_pos.round(3),
                "err:", np.linalg.norm(target_pos - actual_pos).round(3),
                "tau_norm:", np.linalg.norm(tau).round(3),
            )

            viewer.sync()

    if csv_writer is not None:
        fh_csv.close()
        print(f"Saved log: {args.log_csv}")

    tracker.close()


if __name__ == "__main__":
    main()
