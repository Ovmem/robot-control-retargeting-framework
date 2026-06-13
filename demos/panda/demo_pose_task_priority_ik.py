import mujoco
import numpy as np

from scipy.spatial.transform import Rotation


model = mujoco.MjModel.from_xml_path(
    "models/panda/panda.xml"
)

data = mujoco.MjData(model)

# 初始姿态
data.qpos[:7] = np.array([
    0.5,
    -1.0,
    0.8,
    -2.0,
    0.5,
    1.5,
    0.0
])

mujoco.mj_forward(model, data)

body_id = model.body(
    "hand"
).id

# 目标位置
target_pos = np.array([
    0.5,
    0.2,
    0.5
])

# 目标姿态
target_rot = Rotation.from_euler(
    "xyz",
    [0, 0, 0],
    degrees=True
).as_matrix()

dt = 0.01

Kp_pos = 2.0
Kp_rot = 2.0

# 舒适姿态
q_center = (
    model.jnt_range[:7,0]
    +
    model.jnt_range[:7,1]
) / 2

q_ref = q_center

for step in range(3000):

    mujoco.mj_forward(
        model,
        data
    )

    q = data.qpos[:7].copy()

    # ------------------
    # 当前位姿
    # ------------------

    pos = data.xpos[
        body_id
    ].copy()

    R = data.xmat[
        body_id
    ].reshape(3,3)

    # ------------------
    # Position Error
    # ------------------

    pos_error = (
        target_pos
        - pos
    )

    # ------------------
    # Orientation Error
    # ------------------

    R_err = (
        target_rot
        @
        R.T
    )

    rotvec = Rotation.from_matrix(
        R_err
    ).as_rotvec()

    # ------------------
    # Task Velocity
    # ------------------

    v = np.hstack([
        Kp_pos * pos_error,
        Kp_rot * rotvec
    ])

    # ------------------
    # Full Jacobian
    # ------------------

    Jp = np.zeros(
        (3, model.nv)
    )

    Jr = np.zeros(
        (3, model.nv)
    )

    mujoco.mj_jacBody(
        model,
        data,
        Jp,
        Jr,
        body_id
    )

    J = np.vstack([
        Jp[:, :7],
        Jr[:, :7]
    ])

    # ------------------
    # Primary Task
    # ------------------

    J_pinv = np.linalg.pinv(J)

    qdot_primary = (
        J_pinv @ v
    )

    # ------------------
    # Null Space
    # ------------------

    N = (
        np.eye(7)
        -
        J_pinv @ J
    )

    qdot_null = -0.2 * (q - q_ref)

    qdot = (
        qdot_primary
        +
        N @ qdot_null
    )

    data.qpos[:7] += (
        qdot * dt
    )

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
            np.degrees(
                np.linalg.norm(
                    rotvec
                )
            )
        )

        print()

        print(
            "distance to joint center =",
            np.linalg.norm(q - q_center)
        )