import mujoco
import mujoco.viewer

import numpy as np

from scipy.spatial.transform import Rotation


# ------------------------
# 加载模型
# ------------------------

model = mujoco.MjModel.from_xml_path(
    "models/arm6dof.xml"
)

data = mujoco.MjData(model)

ee_id = model.site("ee").id


# ------------------------
# IK参数
# ------------------------

Kp_pos = 2.0
Kp_rot = 2.0

dt = model.opt.timestep


# ------------------------
# 目标位置
# ------------------------

target_pos = np.array([
    0.8,
    0.2,
    -0.3
])


# ------------------------
# 目标姿态
# Roll Pitch Yaw (deg)
# ------------------------

target_R = Rotation.from_euler(
    "xyz",
    [
        20,
        10,
        30
    ],
    degrees=True
).as_matrix()


# ------------------------
# Viewer
# ------------------------

with mujoco.viewer.launch_passive(
    model,
    data
) as viewer:

    step = 0

    while viewer.is_running():

        mujoco.mj_forward(
            model,
            data
        )

        # ====================
        # 当前末端位置
        # ====================

        ee_pos = data.site_xpos[
            ee_id
        ].copy()

        # ====================
        # 当前末端姿态
        # ====================

        R_current = (
            data.site_xmat[
                ee_id
            ]
            .reshape(3, 3)
            .copy()
        )

        # ====================
        # Position Error
        # ====================

        pos_error = (
            target_pos
            - ee_pos
        )

        # ====================
        # Orientation Error
        # ====================

        R_err = (
            target_R
            @ R_current.T
        )

        rotvec = (
            Rotation
            .from_matrix(R_err)
            .as_rotvec()
        )

        # ====================
        # Pose Error
        # ====================

        pose_error = np.hstack([
            pos_error,
            rotvec
        ])

        # ====================
        # Desired Twist
        # ====================

        twist = np.hstack([
            Kp_pos * pos_error,
            Kp_rot * rotvec
        ])

        # ====================
        # Jacobian
        # ====================

        jacp = np.zeros(
            (3, model.nv)
        )

        jacr = np.zeros(
            (3, model.nv)
        )

        mujoco.mj_jacSite(
            model,
            data,
            jacp,
            jacr,
            ee_id
        )

        J = np.vstack([
            jacp,
            jacr
        ])

        # ====================
        # IK
        # ====================

        qdot = (
            np.linalg.pinv(J)
            @ twist
        )

        # ====================
        # 积分
        # ====================

        data.qpos[:] += (
            qdot * dt
        )

        mujoco.mj_forward(
            model,
            data
        )

        viewer.sync()

        # ====================
        # 输出
        # ====================

        if step % 100 == 0:

            print()

            print(
                "step =",
                step
            )

            print()

            print(
                "position error =",
                np.linalg.norm(
                    pos_error
                )
            )

            print()

            print(
                "orientation error (deg) =",
                np.linalg.norm(
                    rotvec
                )
                * 180
                / np.pi
            )

            J_pinv = np.linalg.pinv(J)

            N = np.eye(model.nv) - J_pinv @ J

            print(np.linalg.norm(N))

        # ====================
        # 收敛判断
        # ====================

        if (
            np.linalg.norm(pos_error)
            < 1e-4
            and
            np.linalg.norm(rotvec)
            < np.deg2rad(0.1)
        ):

            print()

            print(
                "Pose IK Converged"
            )

            print()

            print(
                "final position ="
            )

            print(
                ee_pos
            )

            print()

            print(
                "target position ="
            )

            print(
                target_pos
            )

            print()

            print(
                "final joint ="
            )

            print(
                data.qpos.copy()
            )

            break

        step += 1