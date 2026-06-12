import numpy as np

from scipy.spatial.transform import Rotation

# 当前姿态
R_current = Rotation.from_euler(
    "xyz",
    [10,20,30],
    degrees=True
).as_matrix()

# 目标姿态
R_target = Rotation.from_euler(
    "xyz",
    [20,10,40],
    degrees=True
).as_matrix()

# 姿态误差
R_err = R_target @ R_current.T

print("R_err =")
print(R_err)

print()

euler_err = Rotation.from_matrix(
    R_err
).as_euler(
    "xyz",
    degrees=True
)

print("orientation error (deg) =")
print(euler_err)