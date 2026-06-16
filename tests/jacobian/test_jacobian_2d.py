import pytest
import numpy as np

from core.kinematics import jacobian


def test_test_jacobian_2d():
    
    q = np.array([
        0.0,
        0.0
    ])
    
    J = jacobian(q)
    
    J = jacobian(q)
    assert J.shape == (2, 2), f"J.shape={J.shape}"
    assert np.all(np.isfinite(J))
    
    print(J)
    
    # 希望末端向右移动
    
    xdot_des = np.array([
        0.0,
        0.05
    ])
    
    qdot = np.linalg.pinv(J) @ xdot_des
    
    print("qdot:")
    print(qdot)
    assert np.all(np.isfinite(qdot))
if __name__ == "__main__":
    test_test_jacobian_2d()
