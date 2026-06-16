import pytest
import mujoco
import numpy as np


def test_test_ee_pose_6dof():
    
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
    
    print("position =")
    print(data.site_xpos[ee_id])
    
    print()
    
    print("rotation matrix =")
    print(
        data.site_xmat[ee_id].reshape(3,3)
    )
    
    #验证末端x,y,z轴是否正交
    R = data.site_xmat[ee_id].reshape(3,3)
    
    x = R[:,0]
    y = R[:,1]
    z = R[:,2]
    
    print("x·y =", x @ y)
    print("x·z =", x @ z)
    print("y·z =", y @ z)
    assert abs(x @ y) < 1e-10
    assert abs(x @ z) < 1e-10
    assert abs(y @ z) < 1e-10
if __name__ == "__main__":
    test_test_ee_pose_6dof()
