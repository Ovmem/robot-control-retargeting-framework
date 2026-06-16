import pytest
import mujoco


def test_test_body_list():
    
    model = mujoco.MjModel.from_xml_path(
        "models/panda/panda.xml"
    )
    
    print()
    
    print("number of bodies =")
    
    print(model.nbody)
    assert model.nbody > 0
    
    print()
    
    for i in range(model.nbody):
    
        print(
            i,
            model.body(i).name
        )
        assert i >= 0
if __name__ == "__main__":
    test_test_body_list()
