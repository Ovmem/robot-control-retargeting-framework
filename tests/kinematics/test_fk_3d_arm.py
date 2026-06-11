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

from core.kinematics_3d import fk_3d_arm

q = np.array([
    0.5,
    0.3,
    -0.2
])

p0,p1,p2,p3 = fk_3d_arm(q)

print("base =",p0)
print("joint1 =",p1)
print("joint2 =",p2)
print("ee =",p3)