import mujoco

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