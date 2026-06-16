import pytest

pytestmark = pytest.mark.viewer
import mujoco
import mujoco.viewer

def test_test_view_panda():
    """Viewer validation - requires display."""
    pytest.skip("requires viewer GUI and display")
    
    model = mujoco.MjModel.from_xml_path(
        "models/panda/panda.xml"
    )
    
    data = mujoco.MjData(model)
    
    with mujoco.viewer.launch_passive(
        model,
        data
    ) as viewer:
    
        while viewer.is_running():
    
            mujoco.mj_step(
                model,
                data
            )
    
            viewer.sync()
