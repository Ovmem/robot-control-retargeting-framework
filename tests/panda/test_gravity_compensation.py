import mujoco
import mujoco.viewer
import numpy as np
import time


# ==========================
# Load Model
# ==========================

model = mujoco.MjModel.from_xml_path(
    r"models/panda/panda.xml"
)

data = mujoco.MjData(model)

# ==========================
# Initial Joint
# ==========================

data.qpos[:7] = np.array([
    0.0,
    -0.8,
    0.0,
    -2.0,
    0.0,
    2.0,
    0.7
])

mujoco.mj_forward(model, data)

# ==========================
# Viewer
# ==========================

with mujoco.viewer.launch_passive(
    model,
    data
) as viewer:

    step = 0

    while viewer.is_running():

        # 计算动力学项
        mujoco.mj_rnePostConstraint(
            model,
            data
        )

        # 重力补偿
        tau = data.qfrc_bias[:7].copy()

        data.ctrl[:7] = tau

        mujoco.mj_step(
            model,
            data
        )

        if step % 100 == 0:

            print()

            print(
                f"step = {step}"
            )

            print()

            print(
                "gravity torque ="
            )

            print(
                tau
            )

        viewer.sync()

        time.sleep(0.002)

        step += 1