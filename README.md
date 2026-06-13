# 机器人运动控制与动作映射框架

基于 MuJoCo 的机器人运动控制与动作映射框架，实现了从机器人运动学、逆运动学、冗余自由度控制到可操作度优化等核心算法，并为后续人体动作映射（Motion Retargeting）与机器人控制算法验证提供统一仿真平台。

## 项目特点

* 正运动学（Forward Kinematics）
* 雅可比矩阵计算（Jacobian）
* 位置逆运动学（Position IK）
* 位姿逆运动学（Pose IK）
* 四元数与旋转向量姿态表示
* 冗余机械臂 Null Space 控制
* Task Priority 任务优先级控制
* Manipulability 可操作度优化
* MuJoCo 机器人仿真验证

## 当前支持机器人

* Franka Panda 7自由度机械臂

## 技术栈

* Python
* NumPy
* SciPy
* MuJoCo
* Git

## 项目目标

构建面向机器人运控与具身智能的数据采集验证平台，逐步扩展动力学控制、轨迹规划、人体动作映射（Motion Retargeting）等功能。
