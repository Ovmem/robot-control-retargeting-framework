import mujoco
import numpy as np

from scipy.spatial.transform import Rotation

model = mujoco.MjModel.from_xml_path(
    "models/panda/panda.xml"
)

data = mujoco.MjData(model)

mujoco.mj_forward(
    model,
    data
)

body_id = model.body(
    "hand"
).id

R_current = data.xmat[
    body_id
].reshape(3,3)

R_target = Rotation.from_euler(
    "xyz",
    [20, -10, 30],
    degrees=True
).as_matrix()

R_err = (
    R_target
    @
    R_current.T
)

rotvec = Rotation.from_matrix(
    R_err
).as_rotvec()

print()

print("rotation vector =")
print(rotvec)

print()

print(
    "angle (deg) ="
)

print(
    np.degrees(
        np.linalg.norm(rotvec)
    )
)

axis = (
    rotvec
    /
    np.linalg.norm(rotvec)
)

print()

print("rotation axis =")
print(axis)