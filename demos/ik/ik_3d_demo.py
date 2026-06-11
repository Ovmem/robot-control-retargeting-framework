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

from core.kinematics_3d import fk_3d_arm
from core.jacobian_3d import numerical_jacobian

# ----------------------
# DLS Pseudoinverse
# ----------------------

def damped_pinv(J, lam=0.05):

    m = J.shape[0]

    return (
        J.T
        @ np.linalg.inv(
            J @ J.T
            + lam**2 * np.eye(m)
        )
    )

# ----------------------
# target
# ----------------------

target = np.array([
    0.8,
    0.2,
    -0.3
])

# 初始关节角

q = np.array([
    0.5,
    0.3,
    -0.2
])

dt = 0.01

Kp = 5.0

for step in range(1000):

    ee = fk_3d_arm(q)[-1]

    error = target - ee

    J = numerical_jacobian(q)

    xdot = Kp * error

    qdot = damped_pinv(J) @ xdot

    qdot = np.clip(
        qdot,
        -2.0,
        2.0
    )

    q += qdot * dt

    if step % 100 == 0:

        print()

        print("step =", step)

        print("ee =", ee)

        print("target =", target)

        print(
            "error =",
            np.linalg.norm(error)
        )

print()

print("final q =")
print(q)

print()

print("final ee =")
print(fk_3d_arm(q)[-1])