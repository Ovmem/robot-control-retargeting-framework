import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path(
    "models/arm6dof.xml"
)

data = mujoco.MjData(model)

data.qpos[:] = np.array([
    0.3,
    0.4,
    -0.2,
    0.5,
    0.1,
    -0.3
])

mujoco.mj_forward(model, data)

ee_id = model.site("ee").id

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

print("Position Jacobian")
print(jacp)

print()

print("Rotation Jacobian")
print(jacr)

print()

J = np.vstack([
    jacp,
    jacr
])

print("Full Jacobian")
print(J)

print()

print("shape =", J.shape)