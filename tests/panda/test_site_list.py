import pytest
import mujoco


def test_test_site_list():
    
    model = mujoco.MjModel.from_xml_path(
        "models/panda/panda.xml"
    )
    
    print()
    
    print("number of sites =")
    
    print(model.nsite)
    assert model.nsite >= 0
    
    print()
    
    for i in range(model.nsite):
    
        print(
            i,
            model.site(i).name
        )
if __name__ == "__main__":
    test_test_site_list()
