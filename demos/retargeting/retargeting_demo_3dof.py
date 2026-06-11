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
import matplotlib.pyplot as plt

from core.kinematics import (
    fk_3dof,
    jacobian_3dof
)

# 初始姿态

q = np.array([
    0.5,
    0.3,
    -0.2
])

dt = 0.01
Kp = 5.0

human_traj = []
robot_traj = []

for step in range(2000):

    t = step * dt

    # ------------------
    # Human Hand
    # ------------------

    human_hand = np.array([
        0.7 + 0.1*np.cos(t),
        0.2 + 0.1*np.sin(t)
    ])

    ee = fk_3dof(q)

    error = human_hand - ee

    xdot = Kp * error

    J = jacobian_3dof(q)

    qdot = np.linalg.pinv(J) @ xdot

    q += qdot * dt

    human_traj.append(
        human_hand.copy()
    )

    robot_traj.append(
        ee.copy()
    )

    if step % 200 == 0:

        print()

        print("step =", step)

        print("human =", human_hand)

        print("robot =", ee)

        print(
            "tracking error =",
            np.linalg.norm(error)
        )

human_traj = np.array(human_traj)
robot_traj = np.array(robot_traj)

#人体和机械臂末端轨迹图
plt.figure(figsize=(7,7))

plt.plot(
    human_traj[:,0],
    human_traj[:,1],
    label="Human Hand"
)

plt.plot(
    robot_traj[:,0],
    robot_traj[:,1],
    label="Robot EE"
)

plt.xlabel("X (m)")
plt.ylabel("Y (m)")

plt.title(
    "Retargeting Demo"
)

plt.axis("equal")

plt.legend()

plt.grid()

plt.savefig(
    "results/retargeting_demo.png"
)

plt.show()