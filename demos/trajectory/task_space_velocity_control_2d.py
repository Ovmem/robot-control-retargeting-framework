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

from core.kinematics import fk
from core.kinematics import jacobian

model = mujoco.MjModel.from_xml_path(
    "models/arm2d.xml"
)

data = mujoco.MjData(model)
kp = 30
kd = 8
q_target = np.array([
    0.3,
    0.5
])
with mujoco.viewer.launch_passive(
    model,
    data
) as viewer:

    while (
        viewer.is_running()
        and data.time < 10
    ):
        q = data.qpos.copy()

        qd = data.qvel.copy()

        dt = model.opt.timestep
        xdot_des = np.array([
            0.05,
            0.0
        ])
        J = jacobian(q)

        qdot_cmd = (
            np.linalg.pinv(J)
            @ xdot_des
        )

        q_target += qdot_cmd * dt

        tau = (
            kp * (q_target - q)
            - kd * qd
        )

        data.ctrl[:] = tau
        mujoco.mj_step(
            model,
            data
        )
        viewer.sync()
        if int(data.time/dt) % 500 == 0:

            ee = fk(q)

            print()

            print("joint =", q)

            print("ee =", ee)

            print("q_target =", q_target)

            print(
            "det(J) =",
            np.linalg.det(J)
            )

            print("----------------")
