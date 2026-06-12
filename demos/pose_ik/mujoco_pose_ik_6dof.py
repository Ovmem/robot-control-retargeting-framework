import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path(
    "models/arm6dof.xml"
)

data = mujoco.MjData(model)

target = np.array([
    0.8,
    0.2,
    -0.3
])

Kp = 2.0
dt = model.opt.timestep

ee_id = model.site("ee").id

with mujoco.viewer.launch_passive(
    model,
    data
) as viewer:

    step = 0

    while viewer.is_running():

        mujoco.mj_forward(
            model,
            data
        )

        # 当前末端位置
        ee = data.site_xpos[
            ee_id
        ].copy()

        # 位置误差
        pos_error = (
            target - ee
        )

        # 期望末端速度
        v_des = (
            Kp * pos_error
        )

        jacp = np.zeros(
            (3, model.nv)
        )

        jacr = np.zeros(
            (3, model.nv)
        )

        mujoco.mj_jacSite(
            model,
            data,
            jacp,
            jacr,
            ee_id
        )

        qdot = (
            np.linalg.pinv(jacp)
            @ v_des
        )

        data.qpos[:] += (
            qdot * dt
        )

        mujoco.mj_forward(
            model,
            data
        )

        viewer.sync()

        if step % 100 == 0:

            print()

            print(
                "step =",
                step
            )

            print(
                "ee =",
                ee
            )

            print(
                "target =",
                target
            )

            print(
                "error =",
                np.linalg.norm(
                    pos_error
                )
            )

        step += 1

        if np.linalg.norm(pos_error) < 1e-4:

            print("IK Converged")

            break