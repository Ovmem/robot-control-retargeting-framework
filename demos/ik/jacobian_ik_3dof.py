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


import numpy as np

from core.kinematics import (
    fk_3dof,
    jacobian_3dof
)

q = np.array([
    0.5,
    0.3,
    -0.2
])

target = np.array([
    0.8,
    0.2
])

dt = 0.01

Kp = 2.0

for step in range(2000):

    ee = fk_3dof(q)

    error = target - ee

    xdot = Kp * error

    J = jacobian_3dof(q)

    qdot = np.linalg.pinv(J) @ xdot

    q += qdot * dt

    if step % 100 == 0:

        print()

        print("step =", step)

        print("ee =", ee)

        print("target =", target)

        print("error =", np.linalg.norm(error))