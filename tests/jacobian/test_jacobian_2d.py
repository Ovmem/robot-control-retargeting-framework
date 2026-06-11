import numpy as np

from core.kinematics import jacobian

q = np.array([
    0.0,
    0.0
])

J = jacobian(q)

print(J)

# 希望末端向右移动

xdot_des = np.array([
    0.0,
    0.05
])

qdot = np.linalg.pinv(J) @ xdot_des

print("qdot:")
print(qdot)