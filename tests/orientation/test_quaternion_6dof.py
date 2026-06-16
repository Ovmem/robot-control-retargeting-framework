import pytest
import mujoco
import numpy as np

from scipy.spatial.transform import Rotation


def test_test_quaternion_6dof():
    
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
    
    R = data.site_xmat[ee_id].reshape(3,3)
    
    quat = Rotation.from_matrix(R).as_quat()
    
    print("quaternion =")
    print(quat)
    
    print()
    
    print("norm =")
    print(np.linalg.norm(quat))
    assert abs(np.linalg.norm(quat) - 1.0) < 1e-10
if __name__ == "__main__":
    test_test_quaternion_6dof()
