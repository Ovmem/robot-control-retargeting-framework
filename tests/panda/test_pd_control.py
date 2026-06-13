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

q_init = np.array([
    0.5,
    -1.0,
    0.5,
    -2.0,
    0.5,
    2.0,
    1.0
])

data.qpos[:7] = q_init

mujoco.mj_forward(model, data)

# ==========================
# Target Joint
# ==========================

q_des = np.array([
    0.0,
    -0.5,
    0.0,
    -2.0,
    0.0,
    2.0,
    0.7
])

# ==========================
# PD Gain
# ==========================

kp = np.array([
    100,
    100,
    100,
    100,
    100,
    100,
    100
])

kd = np.array([
    20,
    20,
    20,
    20,
    20,
    20,
    20
])

# ==========================
# Viewer
# ==========================

with mujoco.viewer.launch_passive(
    model,
    data
) as viewer:

    step = 0

    while viewer.is_running():

        q = data.qpos[:7].copy()

        qvel = data.qvel[:7].copy()

        # ------------------
        # PD Control
        # ------------------

        tau = (
            kp * (q_des - q)
            +
            kd * (0.0 - qvel)
        )

        # Panda前7个actuator
        data.ctrl[:7] = tau

        mujoco.mj_step(model, data)

        if step % 100 == 0:

            joint_error = np.linalg.norm(
                q_des - q
            )

            print()

            print(
                f"step = {step}"
            )

            print()

            print(
                "joint error ="
            )

            print(
                joint_error
            )

        viewer.sync()

        time.sleep(0.002)

        step += 1