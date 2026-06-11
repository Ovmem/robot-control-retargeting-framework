import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import numpy as np
import pygame

import mujoco
import mujoco.viewer

from core.kinematics_3d import fk_3d_arm
from core.jacobian_3d import numerical_jacobian


# =====================================
# DLS
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
# pygame
# =====================================

pygame.init()

screen = pygame.display.set_mode(
    (300, 200)
)

pygame.display.set_caption(
    "3D Target Controller"
)


# =====================================
# mujoco
# =====================================

model = mujoco.MjModel.from_xml_path(
    "models/arm3d.xml"
)

data = mujoco.MjData(model)

target_body_id = (
    model.body("target").id
)

# =====================================
# state
# =====================================

q = np.array([
    0.5,
    0.3,
    -0.2
])

target = np.array([
    0.8,
    0.2,
    -0.3
])

dt = 0.01

Kp = 5.0

target_speed = 0.4


# =====================================
# viewer
# =====================================

with mujoco.viewer.launch_passive(
    model,
    data
) as viewer:

    while viewer.is_running():

        # -------------------
        # pygame event
        # -------------------

        pygame.event.pump()

        keys = pygame.key.get_pressed()

        # x

        if keys[pygame.K_a]:
            target[0] -= target_speed * dt

        if keys[pygame.K_d]:
            target[0] += target_speed * dt

        # y

        if keys[pygame.K_q]:
            target[1] += target_speed * dt

        if keys[pygame.K_e]:
            target[1] -= target_speed * dt

        # z

        if keys[pygame.K_w]:
            target[2] += target_speed * dt

        if keys[pygame.K_s]:
            target[2] -= target_speed * dt

        # -------------------
        # update target ball
        # -------------------

        model.body_pos[
            target_body_id
        ] = target

        # -------------------
        # IK
        # -------------------

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

        # -------------------
        # update robot
        # -------------------

        data.qpos[:] = q

        mujoco.mj_forward(
            model,
            data
        )

        viewer.sync()

        pygame.display.flip()