import pytest
import mujoco
import numpy as np


def test_test_jacobian():
    
    model = mujoco.MjModel.from_xml_path(
        "models/panda/panda.xml"
    )
    
    data = mujoco.MjData(model)
    
    data.qpos[:7] = np.array([
        0.3,
        -0.5,
        0.2,
        -1.0,
        0.4,
        1.2,
        0.1
    ])
    
    mujoco.mj_forward(
        model,
        data
    )
    
    body_id = model.body(
        "hand"
    ).id
    
    jacp = np.zeros(
        (3, model.nv)
    )
    
    jacr = np.zeros(
        (3, model.nv)
    )
    
    mujoco.mj_jacBody(
        model,
        data,
        jacp,
        jacr,
        body_id
    )
    
    print()
    
    print("Position Jacobian")
    print(jacp)
    
    print()
    
    print("Rotation Jacobian")
    print(jacr)
    
    print()
    
    J = np.vstack([
        jacp,
        jacr
    ])
    
    print("Full Jacobian")
    print(J)
    
    print()
    
    print("shape =", J.shape)
    assert J.shape == (6, model.nv)
if __name__ == "__main__":
    test_test_jacobian()
