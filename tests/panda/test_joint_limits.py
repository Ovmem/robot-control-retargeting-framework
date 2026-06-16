import pytest
import mujoco
import numpy as np


def test_test_joint_limits():
    
    model = mujoco.MjModel.from_xml_path(
        "models/panda/panda.xml"
    )
    
    for i in range(7):
    
        print()
    
        print(
            model.joint(i).name
        )
    
        print(
            model.jnt_range[i]
        )
    
    q_center = (
        model.jnt_range[:7,0]
        +
        model.jnt_range[:7,1]
    ) / 2
    
    print()
    
    print(
        "q_center ="
    )
    
    print(q_center)
    assert q_center.shape == (7,)
    assert all(np.isfinite(q_center))
if __name__ == "__main__":
    test_test_joint_limits()
