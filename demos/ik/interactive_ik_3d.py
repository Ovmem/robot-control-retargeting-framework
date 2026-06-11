import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import time
import numpy as np

import mujoco
import mujoco.viewer

from core.kinematics_3d import fk_3d_arm
from core.jacobian_3d import numerical_jacobian


# ==========================
# DLS
# ==========================

def damped_pinv(J, lam=0.05):

    m = J.shape[0]

    return (
        J.T
        @ np.linalg.inv(
            J @ J.T +
            lam**2*np.eye(m)
        )
    )


# ==========================
# model
# ==========================

model = mujoco.MjModel.from_xml_path(
    "models/arm3d.xml"
)

data = mujoco.MjData(model)


# ==========================
# initial joint
# ==========================

q = np.array([
    0.5,
    0.3,
    -0.2
])

data.qpos[:] = q

mujoco.mj_forward(
    model,
    data
)


# ==========================
# target
# ==========================

target = np.array([
    0.8,
    0.2,
    -0.3
])

target_body_id = (
    model.body("target").id
)


# ==========================
# control
# ==========================

dt = 0.01

Kp = 5.0

step_size = 0.03


# ==========================
# viewer
# ==========================

with mujoco.viewer.launch_passive(
    model,
    data
) as viewer:

    print()

    print("A/D : x")
    print("Q/E : y")
    print("W/S : z")

    while viewer.is_running():

        # ------------------
        # keyboard
        # ------------------

        cmd = input(
            "command:"
        )

        if cmd == "a":
            target[0] -= step_size

        elif cmd == "d":
            target[0] += step_size

        elif cmd == "q":
            target[1] += step_size

        elif cmd == "e":
            target[1] -= step_size

        elif cmd == "w":
            target[2] += step_size

        elif cmd == "s":
            target[2] -= step_size

        # ------------------
        # update target
        # ------------------

        model.body_pos[
            target_body_id
        ] = target

        # ------------------
        # IK
        # ------------------

        for _ in range(30):

            ee = fk_3d_arm(q)[-1]

            error = target - ee

            J = numerical_jacobian(q)

            xdot = Kp * error

            qdot = (
                damped_pinv(J)
                @ xdot
            )

            qdot = np.clip(
                qdot,
                -2,
                2
            )

            q += qdot * dt

        data.qpos[:] = q

        mujoco.mj_forward(
            model,
            data
        )

        print()

        print("target =", target)

        print("ee =", ee)

        print(
            "error =",
            np.linalg.norm(
                target-ee
            )
        )

        viewer.sync()