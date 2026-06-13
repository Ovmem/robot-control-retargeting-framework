import mujoco
import mujoco.viewer

import numpy as np

from scipy.spatial.transform import Rotation


# =====================================
# Manipulability
# =====================================

def compute_manipulability(model, data):

    body_id = model.body("hand").id

    Jp = np.zeros((3, model.nv))
    Jr = np.zeros((3, model.nv))

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

    JJT = J @ J.T

    w = np.sqrt(
        max(
            np.linalg.det(JJT),
            0.0
        )
    )

    return w


# =====================================
# Numerical Gradient
# =====================================

def manipulability_gradient(
        model,
        data,
        q,
        eps=1e-4
):

    grad = np.zeros(7)

    q_backup = q.copy()

    w0 = compute_manipulability(
        model,
        data
    )

    for i in range(7):

        q_test = q_backup.copy()

        q_test[i] += eps

        data.qpos[:7] = q_test

        mujoco.mj_forward(
            model,
            data
        )

        w1 = compute_manipulability(
            model,
            data
        )

        grad[i] = (w1 - w0) / eps

    data.qpos[:7] = q_backup

    mujoco.mj_forward(
        model,
        data
    )

    return grad


# =====================================
# Load Panda
# =====================================

model = mujoco.MjModel.from_xml_path(
    "models/panda/panda.xml"
)

data = mujoco.MjData(model)

# =====================================
# Initial Pose
# =====================================

q = np.array([
    0.0,
    -0.5,
    0.0,
    -2.0,
    0.0,
    2.0,
    0.5
])

data.qpos[:7] = q

mujoco.mj_forward(
    model,
    data
)

# =====================================
# Target Pose
# =====================================

target_pos = np.array([
    0.55,
    0.20,
    0.40
])

target_rot = Rotation.from_euler(
    "xyz",
    [180, 0, 0],
    degrees=True
).as_matrix()

# =====================================
# Viewer
# =====================================

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

        body_id = model.body(
            "hand"
        ).id

        ee_pos = data.xpos[
            body_id
        ].copy()

        ee_rot = data.xmat[
            body_id
        ].reshape(
            3,
            3
        ).copy()

        # =====================
        # Position Error
        # =====================

        pos_err = (
            target_pos
            - ee_pos
        )

        # =====================
        # Orientation Error
        # =====================

        R_err = (
            target_rot
            @ ee_rot.T
        )

        rotvec = Rotation.from_matrix(
            R_err
        ).as_rotvec()

        # =====================
        # Task Error
        # =====================

        err = np.hstack([
            pos_err,
            rotvec
        ])

        # =====================
        # Jacobian
        # =====================

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

        # =====================
        # Primary IK
        # =====================

        J_pinv = np.linalg.pinv(
            J
        )

        qdot_primary = (
            J_pinv
            @
            (2.0 * err)
        )

        # =====================
        # Null Space
        # =====================

        N = (
            np.eye(7)
            -
            J_pinv @ J
        )

        grad = manipulability_gradient(
            model,
            data,
            data.qpos[:7]
        )

        qdot_secondary = (
            0.5
            *
            grad
        )

        qdot = (
            qdot_primary
            +
            N @ qdot_secondary
        )

        # =====================
        # Integrate
        # =====================

        data.qpos[:7] += (
            qdot * 0.01
        )

        mujoco.mj_forward(
            model,
            data
        )

        # =====================
        # Print
        # =====================

        if step % 100 == 0:

            w = compute_manipulability(
                model,
                data
            )

            print()

            print(
                f"step = {step}"
            )

            print()

            print(
                "position error =",
                np.linalg.norm(
                    pos_err
                )
            )

            print()

            print(
                "orientation error (deg) =",
                np.rad2deg(
                    np.linalg.norm(
                        rotvec
                    )
                )
            )

            print()

            print(
                "manipulability =",
                w
            )

        viewer.sync()

        step += 1