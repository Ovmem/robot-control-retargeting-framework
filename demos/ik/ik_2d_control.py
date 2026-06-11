import mujoco
import mujoco.viewer
import numpy as np

from core.kinematics import fk
from core.kinematics import ik


# 加载模型
model = mujoco.MjModel.from_xml_path(
    "models/arm2d.xml"
)

data = mujoco.MjData(model)


# ====================
# 任务空间目标点
# ====================

target = np.array([
    0.8,
    0.2
])

# IK求解目标关节角
q_des = ik(
    target[0],
    target[1]
)

print("Target EE =", target)
print("IK q_des =", q_des)

# ====================
# PD参数
# ====================

kp = 20
kd = 5

step = 0


with mujoco.viewer.launch_passive(
    model,
    data
) as viewer:

    while viewer.is_running():

        q = data.qpos.copy()

        qd = data.qvel.copy()

        tau = (
            kp * (q_des - q)
            - kd * qd
        )

        data.ctrl[:] = tau

        mujoco.mj_step(
            model,
            data
        )

        if step % 500 == 0:

            ee = fk(q)

            print()
            print("step =", step)
            print("joint =", q)
            print("ee =", ee)

            print(
                "ee error =",
                np.linalg.norm(
                    target - ee
                )
            )

            print("----------------")

        step += 1

        viewer.sync()