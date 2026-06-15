# Robot Control & Hand Retargeting Framework

基于 MuJoCo 和 Franka Panda 的机器人运动控制与手部输入动作映射原型。

本项目面向具身智能机器人运动控制、仿真算法和 Motion Retargeting 相关实习展示，重点不是工业现场部署、嵌入式电机驱动或完整人形机器人系统，而是通过可运行的 MuJoCo 仿真代码展示以下能力：

* 机械臂 FK / IK / Jacobian / Pose IK 基础算法；
* Franka Panda 七自由度机械臂的冗余控制、Null Space、Task Priority 和 Manipulability 分析；
* 关节空间 PD、Gravity Compensation、Task-space PD / Impedance-style control；
* MediaPipe hand input 到 Panda 末端目标位姿的 hand-to-Panda retargeting prototype；
* 实验数据记录、曲线绘制和可复现实验流程整理。

当前 retargeting 部分定位为：**手部视觉输入到 Franka Panda 末端目标位姿的动作映射原型**。它不是完整 VLA 系统，也不是完整人形机器人全身 Motion Retargeting 系统。

## Project Structure

```text
robot_control_retargeting_framework/
├── core/
│   ├── controller.py              # damped pseudo-inverse helper
│   ├── dynamics_control.py         # Panda torque control, gravity compensation, task-space control
│   ├── jacobian_3d.py              # numerical Jacobian for simple 3D arm
│   ├── kinematics.py               # 2D / 3-DoF planar FK, IK, Jacobian
│   ├── kinematics_3d.py            # simple 3D arm FK
│   └── pose_ik.py                  # reserved for pose IK interface, currently empty
│
├── retargeting/
│   └── hand_to_panda.py            # hand landmarks -> Panda target pose / gripper command
│
├── vision/
│   ├── hand_tracker.py             # reusable MediaPipe Hands wrapper
│   └── mediapipe_hand.py           # early standalone MediaPipe demo
│
├── demos/
│   ├── panda/                      # main Panda demos for interview presentation
│   ├── ik/                         # IK demos for simple arms and MuJoCo arms
│   ├── pose_ik/                    # 6D pose IK demos
│   ├── jacobian/                   # Jacobian velocity mapping demos
│   ├── nullspace/                  # simple null-space control demo
│   ├── retargeting/                # hand tracker and simple retargeting demos
│   ├── trajectory/                 # trajectory tracking demos
│   └── visualization/              # model visualization helpers
│
├── models/
│   ├── arm2d.xml
│   ├── arm3d.xml
│   ├── arm3dof.xml
│   ├── arm6dof.xml
│   └── panda/                      # Franka Panda MJCF model and mesh assets
│
├── scripts/
│   └── plot_dynamics_results.py    # plot dynamics-control CSV logs and metrics
│
├── results/
│   ├── dynamics/                   # CSV logs, plots, metrics for control experiments
│   └── *.png                       # earlier demo result figures
│
├── tests/                          # validation scripts, not guaranteed one-command pytest suite
├── requirements.txt
└── README.md
```

## Core Modules

### Kinematics, IK and Jacobian

The repository contains simple-arm FK / IK / Jacobian implementations and MuJoCo-based Panda demos. These scripts are mainly used to explain the control stack step by step before moving to the 7-DoF Panda model:

* FK and IK for 2D / 3-DoF toy arms;
* numerical Jacobian checks;
* 6D pose error using rotation vectors;
* damped pseudo-inverse based IK demos.

### Redundancy Control

Franka Panda has 7 DoF, so it can use redundancy when tracking an end-effector task. The Panda demos cover:

* Null-space projection;
* Task-priority IK;
* joint-limit avoidance;
* manipulability metric and numerical-gradient based optimization.

These are useful for interviews about robot motion control and simulation because they connect Jacobian-based control with redundant manipulator behavior.

### Dynamics and Task-space Control

The main reusable controller lives in `core/dynamics_control.py`. It includes:

* joint-space PD torque control;
* PD + gravity compensation through MuJoCo bias / gravity torque;
* inverse-dynamics torque helper;
* task-space PD / impedance-style control;
* Jacobian-transpose wrench-to-torque mapping;
* optional torque clipping and null-space torque projection.

The current PD vs. PD + Gravity Compensation experiment is primarily a reproducible comparison pipeline. Existing metrics may still depend strongly on gains and model actuator settings, so this repository does **not** claim that PD + Gravity Compensation is always significantly better than PD under the current parameters. Control gains still need tuning.

### Hand Input and Retargeting Prototype

The hand-input pipeline is:

```text
camera image
    -> MediaPipe Hands 21 landmarks
    -> wrist / palm frame / pinch ratio
    -> Panda end-effector target position, target orientation, gripper-width command
    -> task-space torque controller
    -> MuJoCo Panda response
```

Current mapping:

```text
wrist horizontal image motion -> Panda end-effector y motion
wrist vertical image motion   -> Panda end-effector z motion
palm frame                    -> Panda end-effector orientation target
thumb-index distance          -> gripper-width command interface
```

For stability, the current demo focuses mainly on image-plane y / z motion. Depth mapping, orientation calibration and gripper integration are still being tuned.

## Environment

Python 3.10 or 3.11 is recommended.

```bash
pip install -r requirements.txt
```

Dependencies:

```text
mujoco
numpy
scipy
matplotlib
opencv-python
mediapipe
```

Run commands from the repository root. On Windows PowerShell, prefer `python -m ...` for package-style demos.

## Recommended Runs

The three main reproducible commands are:

```bash
python -m demos.panda.demo_joint_pd_gc
```

This runs the joint-space PD vs. PD + Gravity Compensation comparison and writes:

```text
results/dynamics/joint_pd_only.csv
results/dynamics/joint_pd_gc.csv
```

```bash
python -m demos.panda.demo_task_space_impedance
```

This runs task-space PD / impedance-style end-effector tracking and writes:

```text
results/dynamics/task_space_impedance.csv
```

```bash
python scripts/plot_dynamics_results.py
```

This plots the saved CSV files and writes:

```text
results/dynamics/pd_gc_tracking_curve.png
results/dynamics/pd_gc_torque_curve.png
results/dynamics/task_space_impedance_error_curve.png
results/dynamics/metrics.csv
```

Optional real-time hand retargeting demo:

```bash
python -m demos.panda.demo_hand_retargeting_pd_gc
```

This demo requires a working camera, OpenCV GUI windows and MuJoCo viewer support. It starts both the MediaPipe hand window and the MuJoCo simulation viewer.

## Results and Interpretation

The repository includes saved dynamics CSV files and figures under `results/dynamics/`. These files are intended to show the experiment pipeline:

* run a control demo;
* record target, actual state, errors and torques;
* plot tracking curves and summary metrics.

Because the controller gains are still being tuned, treat the current PD / PD + Gravity Compensation curves as a comparison setup rather than a final optimized benchmark.

## Tests and Validation Scripts

`tests/` currently contains validation scripts and exploratory checks for kinematics, Jacobian, Panda model loading, gravity compensation, null-space behavior and orientation errors.

It is **not** yet a polished one-command pytest suite. Some scripts print intermediate values, and some may open MuJoCo viewers. For interview presentation, the recommended reproducible path is the three commands in the `Recommended Runs` section.

## Limitations

Current limitations are explicit:

1. This is not a complete VLA system.
2. This is not a complete full-body humanoid Motion Retargeting system.
3. This is not a full legged-robot reinforcement-learning project.
4. The controller has only been validated in MuJoCo simulation, not on a real robot.
5. Hand retargeting currently maps hand input to Panda end-effector targets, not full human-body motion to a humanoid robot.
6. Monocular depth mapping and palm-orientation calibration are still under tuning.
7. The gripper-width command exists in the retargeting interface, but full gripper closed-loop integration still needs cleanup.

## Project Status

Completed or partially completed:

* Franka Panda MuJoCo model loading;
* FK / IK / Jacobian / Pose IK demos;
* Null Space, Task Priority and Manipulability demos;
* joint-space PD and gravity-compensation comparison pipeline;
* task-space PD / impedance-style control demo;
* MediaPipe hand input wrapper;
* hand-to-Panda end-effector target mapping prototype.

Interview-preparation TODOs:

* tune control gains and refresh dynamics metrics;
* add a short demo video or GIF for the hand retargeting demo;
* separate true automated tests from interactive validation scripts;
* clean empty or thin modules such as `core/pose_ik.py`;
* document camera calibration and coordinate-frame conventions for retargeting;
* clean old run logs such as `MUJOCO_LOG.TXT` before publishing.

