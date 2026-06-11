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

import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt

from core.kinematics import fk
from core.kinematics import ik


# --------------------
# 加载模型
# --------------------

model = mujoco.MjModel.from_xml_path(
    "models/arm2d.xml"
)
print(model.opt.timestep)
data = mujoco.MjData(model)

# --------------------
# PD参数
# --------------------

kp = 15
kd = 5

step = 0
max_steps = 5000
# 用于记录数据（后面画图）
time_log = []
error_log = []

# --------------------
# 仿真
# --------------------

with mujoco.viewer.launch_passive(
    model,
    data
) as viewer:

    while viewer.is_running() and data.time < 20:

        t = data.time

        # ====================
        # 圆轨迹
        # 圆心(0.6,0.2)
        # 半径0.15
        # ====================

        x = (
            0.6
            + 0.15 * np.cos(0.5 * t)
        )

        y = (
            0.2
            + 0.15 * np.sin(0.5 * t)
        )

        target = np.array([
            x,
            y
        ])

        # IK求目标关节角
        q_des = ik(x, y)

        # 当前状态
        q = data.qpos.copy()
        qd = data.qvel.copy()

        # PD控制
        tau = (
            kp * (q_des - q)
            - kd * qd
        )

        data.ctrl[:] = tau

        mujoco.mj_step(
            model,
            data
        )

        # 当前末端位置
        ee = fk(q)

        # 跟踪误差
        error = np.linalg.norm(
            target - ee
        )

        time_log.append(t)
        error_log.append(error)

        if step % 500 == 0:
            window_error = error_log[-1000:]
            avg_error = np.mean(window_error)
            max_error = np.max(window_error)
            rmse = np.sqrt(
                np.mean(
                    np.square(
                        error_log[-1000:]
                    )
                )
            )


            print()
            print("step =", step)

            print(
                "target =",
                target
            )

            print(
                "ee =",
                ee
            )

            print(
                "error =",
                error
            )
            print("avg error =", avg_error)
            print("max error =", max_error)
            print("RMSE =", rmse)
            print("----------------")

        step += 1

        viewer.sync()

# =====================
# 仿真结束后画图
# =====================

plt.figure(figsize=(8,4))

plt.plot(
    time_log,
    error_log
)

plt.xlabel("Time (s)")
plt.ylabel("Tracking Error (m)")
plt.title("End-Effector Tracking Error")

plt.grid()

plt.savefig(
    "tracking_error.png",
    dpi=300
)

plt.show()