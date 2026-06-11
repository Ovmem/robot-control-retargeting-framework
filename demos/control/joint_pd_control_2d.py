import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path(
    "models/arm2d.xml"
)

data = mujoco.MjData(model)



kp = 20
kd = 1
step = 0

with mujoco.viewer.launch_passive(
    model,
    data
) as viewer:

    while viewer.is_running() :
        #当前时间
        t = data.time
        # 目标关节角
        q_des = np.array([
            0.3*np.sin(0.2*t),
            0.15*np.sin(0.2*t)
        ])

        # 当前关节角度
        q = data.qpos.copy()

        # 当前关节角速度
        qd = data.qvel.copy()
        
        # PD控制器
        tau = kp * (q_des - q) - kd * qd
        

        # 输出控制力矩
        data.ctrl[:] = tau
        mujoco.mj_step(model, data)

        error = q_des - q
        # 每500步打印一次
        if step % 500 == 0:
            print("step =", step)
            print("q =", q)
            print("qd =", qd)
            print("tau =", tau)
            print("-------------------")
            print(np.linalg.norm(error))
        step += 1

        viewer.sync()