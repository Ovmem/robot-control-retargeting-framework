# demos/panda/demo_hand_retargeting_pd_gc.py

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import time

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation

from core.dynamics_control import (
    PandaTorqueController,
    TorqueLimit,
    get_body_pose,
)
from retargeting.hand_to_panda import HandToPandaRetargeter
from vision.hand_tracker import MediaPipeHandTracker


def main():
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

    retargeter = HandToPandaRetargeter(
        robot_origin=base_pos.copy(),
        position_scale_xy=2.2,
        depth_scale=0,
        filter_alpha=0.18,
    )

    # Default target before hand is detected.
    target_pos = np.array([0.45, 0.0, 0.45])
    # target_rot = Rotation.from_euler("xyz", [180, 0, 90], degrees=True).as_matrix()

    print("Press ESC in the MediaPipe window or close MuJoCo viewer to exit.")
    print("Move your wrist to command end-effector position.")
    print("Pinch thumb-index to command gripper width, currently printed only.")

    last_print = time.time()

    sim_substeps = 60  # 可以试 10、15、20

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

            for _ in range(sim_substeps):
                    # 如果 XML 里是 position actuator，让它保持当前关节角，避免它把机械臂拉回 ctrl=0
                if model.nu >= 7:
                    data.ctrl[:7] = data.qpos[:7]

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

    tracker.close()


if __name__ == "__main__":
    main()
