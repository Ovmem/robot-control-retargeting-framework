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

from core.jacobian_3d import numerical_jacobian
from core.kinematics_3d import fk_3d_arm

q = np.array([
    0.5,
    0.3,
    -0.2
])

J = numerical_jacobian(q)

print(J)

qdot = np.array([
    0.1,
    0.2,
    -0.1
])

xdot = J @ qdot


dt = 1e-4

q_new = q + qdot * dt

p_old = fk_3d_arm(q)[-1]

p_new = fk_3d_arm(q_new)[-1]

xdot_real = (p_new - p_old)/dt

print("jacobian =", xdot)

print("real =", xdot_real)