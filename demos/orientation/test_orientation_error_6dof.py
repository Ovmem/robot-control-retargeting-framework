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

# 旋转误差
R_err = R_target @ R_current.T

# 转旋转向量
rotvec = Rotation.from_matrix(
    R_err
).as_rotvec()

print("rotation vector =")
print(rotvec)

print()

print("angle (deg) =")
print(
    np.linalg.norm(rotvec)
    * 180
    / np.pi
)