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

from core.kinematics import (
    fk_3dof,
    jacobian_3dof
)

q = np.array([
    0.5,
    0.3,
    -0.2
])

J = jacobian_3dof(q)

print("J =")
print(J)

print()

qdot = np.array([
    0.1,
    0.2,
    0.3
])

xdot = J @ qdot

print("qdot =")
print(qdot)

print()

print("xdot =")
print(xdot)