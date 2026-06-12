import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path(
    "models/arm6dof.xml"
)

data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(
    model,
    data
) as viewer:

    while viewer.is_running():

        mujoco.mj_step(model, data)

        viewer.sync()

print(model.nq)
print(model.nv)
print(model.nu)

site_id = model.site("ee").id
print(site_id)