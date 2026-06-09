# Robot Control Project

A robotics control learning project based on MuJoCo and Python.

This project implements the fundamental algorithms used in robot motion control, including Forward Kinematics, Inverse Kinematics, Jacobian-based differential kinematics, joint-space PD control, and end-effector trajectory tracking.

## Features

* Forward Kinematics (FK)
* Inverse Kinematics (IK)
* Jacobian Matrix Computation
* Joint Space PD Control
* Velocity Control
* End-Effector Circular Trajectory Tracking

## Environment

* Python 3.10
* MuJoCo 3.9
* NumPy
* Matplotlib

## Results

Trajectory Tracking Performance:

* Average Error ≈ 0.024 m
* RMSE ≈ 0.024 m

## Project Structure

```text
robot_control_project
│
├─core
│   └── kinematics.py
│
├─demos
│   ├── pd_control.py
│   ├── ik_control.py
│   └── trajectory_tracking.py
│   ├── velocity_control.py
│
├─models
│      arm2d.xml
│
├─tests
│      test_jacobian.py
│
├─results
│      trajectory_tracking.png
│      tracking_error.png
│
└─__pycache__
        kinematics.cpython-310.pyc
```

## Future Work

* Differential Inverse Kinematics
* Motion Retargeting
* Whole-Body Control
* Humanoid Robot Control
