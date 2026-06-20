# demos/panda/demo_hand_retargeting_pd_gc.py
"""Main experiment: camera -> hand landmarks -> Panda target -> MuJoCo tracking -> data logging."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation

from core.dynamics_control import (
    PandaTorqueController,
    TorqueLimit,
    get_body_pose,
    rotation_error_rotvec,
)
from retargeting.hand_to_panda import HandToPandaRetargeter
from vision.hand_tracker import MediaPipeHandTracker


CSV_FIELDS = [
    "timestamp", "frame_id", "detected_hand", "detection_confidence",
    "wrist_x", "wrist_y", "wrist_z",
    "pinch_ratio",
    "target_pos_x", "target_pos_y", "target_pos_z",
    "filtered_target_pos_x", "filtered_target_pos_y", "filtered_target_pos_z",
    "target_quat_w", "target_quat_x", "target_quat_y", "target_quat_z",
    "gripper_width", "workspace_clipped",
    "actual_ee_pos_x", "actual_ee_pos_y", "actual_ee_pos_z",
    "ee_position_error", "ee_orientation_error",
    "joint_q_1", "joint_q_2", "joint_q_3", "joint_q_4",
    "joint_q_5", "joint_q_6", "joint_q_7",
    "joint_dq_1", "joint_dq_2", "joint_dq_3", "joint_dq_4",
    "joint_dq_5", "joint_dq_6", "joint_dq_7",
    "torque_norm", "max_abs_torque",
]


def make_parser():
    p = argparse.ArgumentParser(
        description="Hand retargeting: camera -> Panda -> data logging")
    p.add_argument("--camera-id", type=int, default=0)
    p.add_argument("--duration", type=float, default=20)
    p.add_argument("--pos-scale", type=float, default=2.2)
    p.add_argument("--filter-alpha", type=float, default=0.18)
    p.add_argument("--output-dir", type=str, default="results/hand_retargeting/runs")
    p.add_argument("--show-camera", action="store_true", help="Show camera window")
    return p


def main():
    args = make_parser().parse_args()

    # --- Output dir ---
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / run_id
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / "hand_retargeting_run.csv"

    # --- MuJoCo setup ---
    model = mujoco.MjModel.from_xml_path("models/panda/panda.xml")
    data = mujoco.MjData(model)

    q_home = np.array([0.0, -0.7, 0.0, -2.2, 0.0, 1.6, 0.8])
    data.qpos[:7] = q_home
    data.qvel[:7] = 0.0
    if model.nu >= 7:
        data.ctrl[:7] = q_home
    mujoco.mj_forward(model, data)

    body_name = "hand"
    base_pos, base_rot = get_body_pose(model, data, body_name)

    torque_limit = TorqueLimit(
        lower=-np.array([87, 87, 87, 87, 12, 12, 12], dtype=float),
        upper=np.array([87, 87, 87, 87, 12, 12, 12], dtype=float),
    )
    controller = PandaTorqueController(
        model=model, data=data, dof=7, body_name=body_name,
        torque_limit=torque_limit,
    )

    # --- Retargeter ---
    retargeter = HandToPandaRetargeter(
        robot_origin=base_pos.copy(),
        position_scale_xy=args.pos_scale,
        depth_scale=0.0,
        enable_depth_mapping=False,
        filter_alpha=args.filter_alpha,
    )

    # --- Camera ---
    try:
        tracker = MediaPipeHandTracker(
            camera_id=args.camera_id, draw=False, mirror=True,
        )
    except RuntimeError as e:
        print(f"Camera error: {e}")
        return

    # --- CSV ---
    fh_csv = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(fh_csv, fieldnames=CSV_FIELDS)
    csv_writer.writeheader()

    print(f"\n=== Hand Retargeting Demo ===")
    print(f"  Run ID:    {run_id}")
    print(f"  Duration:  {args.duration}s")
    print(f"  Pos scale: {args.pos_scale}")
    print(f"  Filter α:  {args.filter_alpha}")
    print(f"  Output:    {csv_path}")
    print(f"  Camera:    {'window' if args.show_camera else 'no display'}")
    print("==============================\n")

    target_pos = base_pos.copy()
    target_rot = base_rot.copy()
    target_gripper = 0.04
    sim_substeps = 60

    if args.show_camera:
        cv2.namedWindow("Hand Retargeting Camera", cv2.WINDOW_NORMAL)

    demo_start = time.time()
    frame_id = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            elapsed = time.time() - demo_start
            if elapsed > args.duration:
                print(f"Reached {args.duration}s.")
                break

            obs = tracker.read()
            detected = obs is not None
            score = obs.score if detected else 0.0
            wrist_pos = obs.landmarks_image[0].copy() if detected else np.zeros(3)

            # --- Single retargeter.update() call ---
            if detected:
                target = retargeter.update(obs)
                if target.valid:
                    target_pos = target.pos
                    target_rot = target.rot
                    target_gripper = target.gripper_width

            # --- Control ---
            tau = controller.task_space_pd(
                target_pos=target_pos, target_rot=target_rot,
                kp_pos=np.array([1400.0, 1100.0, 1600.0]),
                kd_pos=np.array([100.0, 85.0, 110.0]),
                kp_rot=8.0, kd_rot=2.0, gravity_comp=True,
            )

            for _ in range(sim_substeps):
                if model.nu >= 7:
                    data.ctrl[:7] = data.qpos[:7]
                if model.nu >= 8:
                    data.ctrl[7] = np.clip(target_gripper / 0.04 * 255.0, 0, 255)
                controller.apply_torque(tau, prefer_ctrl=False)
                mujoco.mj_step(model, data)

            actual_pos, actual_rot = get_body_pose(model, data, "hand")
            pos_err = np.linalg.norm(target_pos - actual_pos)
            rot_err = np.linalg.norm(rotation_error_rotvec(target_rot, actual_rot))
            tau_norm = float(np.linalg.norm(tau))
            target_quat = Rotation.from_matrix(target_rot).as_quat()
            q, qd = data.qpos[:7].copy(), data.qvel[:7].copy()

            # --- CSV ---
            row = {
                "timestamp": f"{elapsed:.3f}", "frame_id": frame_id,
                "detected_hand": int(detected),
                "detection_confidence": f"{score:.4f}" if detected else "",
                "wrist_x": f"{wrist_pos[0]:.6f}", "wrist_y": f"{wrist_pos[1]:.6f}", "wrist_z": f"{wrist_pos[2]:.6f}",
                "pinch_ratio": f"{target.pinch_ratio:.4f}" if detected else "",
                "target_pos_x": f"{target_pos[0]:.4f}", "target_pos_y": f"{target_pos[1]:.4f}", "target_pos_z": f"{target_pos[2]:.4f}",
                "filtered_target_pos_x": f"{target_pos[0]:.4f}", "filtered_target_pos_y": f"{target_pos[1]:.4f}", "filtered_target_pos_z": f"{target_pos[2]:.4f}",
                "target_quat_w": f"{target_quat[3]:.6f}", "target_quat_x": f"{target_quat[0]:.6f}",
                "target_quat_y": f"{target_quat[1]:.6f}", "target_quat_z": f"{target_quat[2]:.6f}",
                "gripper_width": f"{target_gripper:.4f}",
                "workspace_clipped": "",
                "actual_ee_pos_x": f"{actual_pos[0]:.4f}", "actual_ee_pos_y": f"{actual_pos[1]:.4f}", "actual_ee_pos_z": f"{actual_pos[2]:.4f}",
                "ee_position_error": f"{pos_err:.4f}", "ee_orientation_error": f"{rot_err:.6f}",
                "torque_norm": f"{tau_norm:.2f}", "max_abs_torque": f"{float(np.max(np.abs(tau))):.2f}",
            }
            for i in range(7):
                row[f"joint_q_{i+1}"] = f"{q[i]:.4f}"
                row[f"joint_dq_{i+1}"] = f"{qd[i]:.4f}"
            csv_writer.writerow(row)
            frame_id += 1

            # --- Camera window ---
            if args.show_camera and obs is not None:
                frame = obs.frame_bgr.copy()
                y = 30
                for text in [
                    f"Detected: {score:.2f}",
                    f"Target: ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})",
                    f"EE err: {pos_err:.3f}m",
                    f"Gripper: {target_gripper:.4f}m",
                    "ESC to quit",
                ]:
                    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                    y += 28
                cv2.imshow("Hand Retargeting Camera", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            print(f"frame={frame_id} err={pos_err:.3f}m tau={tau_norm:.1f}Nm", flush=True)

    fh_csv.close()
    tracker.close()
    cv2.destroyAllWindows()
    print(f"\nSaved: {csv_path}")
    print(f"Next: python scripts/analyze_hand_retargeting_run.py --input {csv_path}")


if __name__ == "__main__":
    main()
