import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path(
    "models/panda/panda.xml"
)

data = mujoco.MjData(model)

# 当前姿态
q = np.array([
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0
])

data.qpos[:7] = q

mujoco.mj_forward(
    model,
    data
)

body_id = model.body(
    "hand"
).id

Jp = np.zeros(
    (3, model.nv)
)

Jr = np.zeros(
    (3, model.nv)
)

mujoco.mj_jacBody(
    model,
    data,
    Jp,
    Jr,
    body_id
)

J = np.vstack([
    Jp[:, :7],
    Jr[:, :7]
])

JJT = J @ J.T

w = np.sqrt(
    np.linalg.det(
        JJT
    )
)

print()

print("rank =")
print(
    np.linalg.matrix_rank(J)
)

print()

print("manipulability =")
print(w)

U,S,Vt = np.linalg.svd(J)

print()
print("singular values =")
print(S)