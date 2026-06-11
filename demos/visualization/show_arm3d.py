import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path(
    "models/arm3d.xml"
)

data = mujoco.MjData(model)

data.qpos[:] = [
    1.2,
    0.8,
    -0.5
]

mujoco.mj_forward(
    model,
    data
)

with mujoco.viewer.launch_passive(
    model,
    data
) as viewer:

    while viewer.is_running():

        viewer.sync()  