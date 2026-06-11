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

J = jacobian_3dof(q)
   
print("J =")
print(J)

print()

xdot_des = np.array([
    0.1,
    0.0
])

print("desired ee velocity =")
print(xdot_des)

print()

qdot = np.linalg.pinv(J) @ xdot_des

print("qdot =")
print(qdot)

print()

xdot_real = J @ qdot

print("real ee velocity =")
print(xdot_real)