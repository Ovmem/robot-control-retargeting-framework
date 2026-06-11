import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )
    )
)

import time
import numpy as np

import mujoco
import mujoco.viewer

from core.kinematics_3d import fk_3d_arm
from core.jacobian_3d import numerical_jacobian


# =====================================
# Damped Least Squares
# =====================================

def damped_pinv(J, lam=0.05):

    m = J.shape[0]

    return (
        J.T
        @ np.linalg.inv(
            J @ J.T
            + lam**2 * np.eye(m)
        )
    )


# =====================================
# Load Model
# =====================================

model = mujoco.MjModel.from_xml_path(
    "models/arm3d.xml"
)

data = mujoco.MjData(model)


# =====================================
# Initial Joint State
# =====================================

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


# =====================================
# Target Position
# =====================================

target = np.array([
    0.8,
    0.2,
    -0.3
])

print("Target =", target)


# =====================================
# Controller Parameters
# =====================================

Kp = 5.0

dt = 0.01


# =====================================
# Viewer
# =====================================

with mujoco.viewer.launch_passive(
    model,
    data
) as viewer:

    step = 0

    while viewer.is_running():

        # -------------------------
        # FK
        # -------------------------

        ee = fk_3d_arm(q)[-1]

        # -------------------------
        # Position Error
        # -------------------------

        error = target - ee

        # -------------------------
        # Jacobian
        # -------------------------

        J = numerical_jacobian(q)

        # -------------------------
        # Cartesian Velocity
        # -------------------------

        xdot = Kp * error

        # -------------------------
        # DLS IK
        # -------------------------

        qdot = (
            damped_pinv(J)
            @ xdot
        )

        # -------------------------
        # Joint Velocity Limit
        # -------------------------

        qdot = np.clip(
            qdot,
            -2.0,
            2.0
        )

        # -------------------------
        # Integrate
        # -------------------------

        q += qdot * dt

        # -------------------------
        # Update MuJoCo
        # -------------------------

        data.qpos[:] = q

        mujoco.mj_forward(
            model,
            data
        )

        # -------------------------
        # Debug
        # -------------------------

        if step % 100 == 0:

            print()

            print("step =", step)

            print("ee =", ee)

            print("error =",
                  np.linalg.norm(error))

        step += 1

        viewer.sync()

        time.sleep(dt)