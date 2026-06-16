import pytest
import sys
import os

tests_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(tests_dir))

import numpy as np

from core.kinematics import fk_3dof, jacobian_3dof


def test_jacobian_3dof():
    q = np.array([0.5, 0.3, -0.2])
    J = jacobian_3dof(q)
    assert J.shape == (2, 3)
    assert np.all(np.isfinite(J))
    
    qdot = np.array([0.1, 0.2, 0.3])
    xdot = J @ qdot
    assert xdot.shape == (2,)
    assert np.all(np.isfinite(xdot))


if __name__ == "__main__":
    test_jacobian_3dof()
