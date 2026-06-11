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
import matplotlib.pyplot as plt

from core.kinematics import (
    fk_3dof,
    jacobian_3dof
)

# 初始关节角
q = np.array([
    1.5,
    -2.0,
    1.0
])

# 目标点
target = np.array([
    0.8,
    0.2
])

dt = 0.01
Kp = 5.0

# 记录数据
q_history = []
ee_history = []
ee_error_history = []

for step in range(2000):

    # 当前末端位置
    ee = fk_3dof(q)

    # 主任务：末端到达目标
    error = target - ee

    xdot = Kp * error

    J = jacobian_3dof(q)

    J_pinv = np.linalg.pinv(J)

    # ==========================
    # Null Space 部分
    # ==========================

    N = np.eye(3) - J_pinv @ J

    q_center = np.array([
        0.0,
        0.0,
        0.0
    ])

    k_null = 1.0

    qdot_null = -k_null * (
        q - q_center
    )   

    qdot = (
        J_pinv @ xdot
        +
        N @ qdot_null
    )

    q = q + qdot * dt
    
    ee_error_history.append(
        np.linalg.norm(error)
    )
    q_history.append(q.copy())
    ee_history.append(ee.copy())

    null_norm = np.linalg.norm(
        N @ qdot_null
    )

    if step % 200 == 0:

        print()

        print("null motion =", null_norm)
        print("step =", step)
        print("joint =", q)
        print("ee =", ee)
        print("error =", np.linalg.norm(error))

q_history = np.array(q_history)
ee_history = np.array(ee_history)

#关节变化图
plt.figure(figsize=(8,5))

plt.plot(q_history[:,0], label="q1")
plt.plot(q_history[:,1], label="q2")
plt.plot(q_history[:,2], label="q3")

plt.title("Joint Motion in Null Space")
plt.xlabel("Step")
plt.ylabel("Joint Angle (rad)")

plt.legend()
plt.grid()

plt.savefig("results/nullspace_joint.png")

plt.show()

#末端位置变化图
plt.figure(figsize=(8,5))

plt.plot(ee_history[:,0], label="x")
plt.plot(ee_history[:,1], label="y")

plt.title("End Effector Position")
plt.xlabel("Step")
plt.ylabel("Position (m)")

plt.legend()
plt.grid()

plt.savefig("results/nullspace_ee.png")

plt.show()

#末端位置误差变化图
plt.figure()

plt.plot(ee_error_history)

plt.title("End Effector Error")

plt.grid()

plt.savefig(
    "results/nullspace_error.png"
)

plt.show()
