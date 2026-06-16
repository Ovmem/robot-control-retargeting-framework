import pytest
import mujoco


def test_test_load_panda():
    
    model = mujoco.MjModel.from_xml_path(
        "models/panda/panda.xml"
    )
    
    data = mujoco.MjData(model)
    
    print()
    
    print("nq =", model.nq)
    
    print("nv =", model.nv)
    
    print("nu =", model.nu)
    assert model.nq >= 7
    assert model.nv >= 7
    assert model.nu >= 7
    
    print()
    
    print("number of joints =", model.njnt)
    
    print()
    
    for i in range(model.njnt):
    
        print(
            i,
            model.joint(i).name
        )
if __name__ == "__main__":
    test_test_load_panda()
