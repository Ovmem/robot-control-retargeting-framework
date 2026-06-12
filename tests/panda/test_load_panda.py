import mujoco

model = mujoco.MjModel.from_xml_path(
    "models/panda/panda.xml"
)

data = mujoco.MjData(model)

print()

print("nq =", model.nq)

print("nv =", model.nv)

print("nu =", model.nu)

print()

print("number of joints =", model.njnt)

print()

for i in range(model.njnt):

    print(
        i,
        model.joint(i).name
    )