import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)
import numpy as np

from core.kinematics import fk_3dof


q = np.array([0, 0, 0])

print("q =")
print(q)

print()

print("ee =")
print(fk_3dof(q))