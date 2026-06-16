# Robot Control & Hand Retargeting Framework

基于 MuJoCo 和 Franka Panda 的机器人运动控制与手部输入动作映射原型。

本项目面向“具身智能机器人运动控制 / 仿真算法 / Motion Retargeting 实习”展示，重点不是传统嵌入式、电机驱动、工业机械臂现场调试，也不是把项目包装成完整 VLA、完整人形机器人全身控制或足式机器人强化学习系统。当前目标是通过可运行的 MuJoCo 仿真代码展示：

* 机械臂 FK / IK / Jacobian / Pose IK 基础算法；
* Franka Panda 七自由度机械臂的 Null Space、Task Priority 和 Manipulability 分析；
* 关节空间 PD、Gravity Compensation、Task-space PD / Impedance-style control；
* MediaPipe hand input 到 Panda 末端目标位姿的 hand-to-Panda retargeting prototype；
* 实验数据记录、曲线绘制和可复现实验流程整理。

当前 retargeting 部分定位为：**手部视觉输入到 Franka Panda 末端目标位姿的动作映射原型**。它不是完整 VLA 系统，也不是完整人形机器人全身 Motion Retargeting 系统。

## 项目结构

```text
robot_control_retargeting_framework/
├── core/
│   ├── controller.py              # 阻尼伪逆辅助函数
│   ├── dynamics_control.py         # Panda 力矩控制、重力补偿、任务空间控制
│   ├── jacobian_3d.py              # 简化 3D 机械臂数值雅可比
│   ├── kinematics.py               # 2D / 3-DoF 平面机械臂 FK、IK、Jacobian
│   ├── kinematics_3d.py            # 简化 3D 机械臂 FK
│   └── pose_ik.py                  # 轻量级末端位姿 IK（阻尼最小二乘）
│
├── retargeting/
│   └── hand_to_panda.py            # 手部 landmarks -> Panda 目标位姿 / 夹爪指令
│
├── vision/
│   ├── hand_tracker.py             # 可复用 MediaPipe Hands 封装
│   └── mediapipe_hand.py           # 早期独立 MediaPipe demo
│
├── demos/
│   ├── panda/                      # 面试展示主线：Panda 控制与 retargeting demo
│   ├── ik/                         # 简化机械臂和 MuJoCo 机械臂 IK demo
│   ├── pose_ik/                    # 6D 位姿 IK demo
│   ├── jacobian/                   # Jacobian 速度映射 demo
│   ├── nullspace/                  # 简化 Null Space 控制 demo
│   ├── retargeting/                # 手部检测和简化 retargeting demo
│   ├── trajectory/                 # 轨迹跟踪 demo
│   └── visualization/              # 模型可视化辅助脚本
│
├── models/
│   ├── arm2d.xml
│   ├── arm3d.xml
│   ├── arm3dof.xml
│   ├── arm6dof.xml
│   └── panda/                      # Franka Panda MJCF 模型和 mesh 资源
│
├── scripts/
│   └── plot_dynamics_results.py    # 绘制控制实验 CSV 日志和指标
│
├── results/
│   ├── dynamics/                   # 控制实验 CSV、曲线和 metrics
│   └── *.png                       # 早期 demo 结果图
│
├── tests/                          # 验证脚本集合，不保证一键 pytest 通过
├── requirements.txt
└── README.md
```

## 核心模块

### 运动学、IK 与 Jacobian

仓库包含简化机械臂的 FK / IK / Jacobian 实现，以及基于 MuJoCo 的 Panda demo。这些脚本主要用于把控制链路从基础算法逐步过渡到 7 自由度 Panda：

* 2D / 3-DoF toy arm 的 FK 和 IK；
* 数值 Jacobian 验证；
* 基于旋转向量的 6D 位姿误差；
* 基于伪逆 / 阻尼伪逆的 IK demo。

### 冗余控制

Franka Panda 是 7 自由度机械臂，在完成末端任务时存在冗余自由度。Panda 相关 demo 覆盖：

* Null Space Projection；
* Task Priority IK；
* Joint Limit Avoidance；
* Manipulability 指标和数值梯度优化。

这些内容适合支撑机器人运动控制和仿真算法面试，因为它们把 Jacobian 控制、冗余机械臂行为和构型优化联系在一起。

### 动力学与任务空间控制

主要可复用控制器位于 `core/dynamics_control.py`，包括：

* 关节空间 PD 力矩控制；
* 基于 MuJoCo bias / gravity torque 的 PD + Gravity Compensation；
* inverse dynamics torque 辅助函数；
* Task-space PD / Impedance-style control；
* Jacobian transpose 的 wrench-to-torque 映射；
* 可选力矩裁剪和 Null Space torque projection。

当前 PD 与 PD + Gravity Compensation 实验主要用于建立可复现的对比流程。已有实验发现，标准 Panda MJCF 使用 affine-bias position actuators（general type, dyntype=none, biastype=affine），当 data.ctrl 默认为 0 时，内置位置伺服会与 qfrc_applied 外加力矩对抗。当前已通过令 ``data.ctrl[:7] = data.qpos[:7]`` 的方式中和这一冲突（见 ``neutralize_position_actuators``）。在修正后的验证流程下，PD + Gravity Compensation 相比 PD-only 取得更低跟踪误差。

当前结果仍是 MuJoCo 仿真验证，不构成真机实验结论，也不是最终 torque-actuated Panda XML 基准。

### 手部输入与 Retargeting 原型

手部输入链路如下：

```text
摄像头图像
    -> MediaPipe Hands 21 个手部关键点
    -> wrist / palm frame / pinch ratio
    -> Panda 末端目标位置、目标姿态、夹爪宽度指令
    -> 任务空间力矩控制器
    -> MuJoCo Panda 响应
```

当前映射关系：

```text
手腕在图像中左右移动 -> Panda 末端 y 方向运动
手腕在图像中上下移动 -> Panda 末端 z 方向运动
手掌坐标系           -> Panda 末端目标姿态
拇指-食指距离        -> 夹爪宽度指令接口
```

为了保证演示稳定，当前 demo 主要验证图像平面内的 y / z 运动。深度方向映射、姿态标定和夹爪闭环接入仍在整理中。

坐标帧假设和映射约束详见 ``docs/retargeting.md``。

## 环境配置

推荐使用 Python 3.10 或 3.11。

```bash
pip install -r requirements.txt
```

主要依赖：

```text
mujoco
numpy
scipy
matplotlib
opencv-python
mediapipe
```

请在仓库根目录运行命令。Windows PowerShell 下推荐使用 `python -m ...` 的方式运行 package-style demo。

## 推荐运行命令

三条主线可复现命令：

```bash
python -m demos.panda.demo_joint_pd_gc
```

该命令运行关节空间 PD 与 PD + Gravity Compensation 的对比实验，并输出：

```text
results/dynamics/joint_pd_only.csv
results/dynamics/joint_pd_gc.csv
```

```bash
python -m demos.panda.demo_task_space_impedance
```

该命令运行 task-space PD / impedance-style 末端跟踪实验，并输出：

```text
results/dynamics/task_space_impedance.csv
```

```bash
python scripts/plot_dynamics_results.py
```

该命令读取已保存的 CSV 文件并绘图，输出：

```text
results/dynamics/pd_gc_tracking_curve.png
results/dynamics/pd_gc_torque_curve.png
results/dynamics/task_space_impedance_error_curve.png
results/dynamics/metrics.csv
```

可选实时手部 retargeting demo：

```bash
python -m demos.panda.demo_hand_retargeting_pd_gc
```

该 demo 需要可用摄像头、OpenCV GUI 窗口和 MuJoCo viewer 支持。运行后会同时启动 MediaPipe 手部窗口和 MuJoCo 仿真窗口。

## 结果解释

仓库在 `results/dynamics/` 下保存了 dynamics 实验 CSV、曲线图和 metrics。这些文件主要用于展示实验流程：

* 运行控制 demo；
* 记录目标、实际状态、误差和力矩；
* 绘制 tracking 曲线和 summary metrics。

由于控制器增益仍在整定，当前 PD / PD + Gravity Compensation 曲线应被理解为“对比实验流程”，不是最终优化 benchmark。

### Dynamics Metrics

在修正 Panda position actuator 与 qfrc_applied 的接口冲突后，当前保存结果如下：

| Metric | Value |
|---|---|
| PD-only mean joint error | 0.2536 rad |
| PD+GC mean joint error | 0.2383 rad |
| PD-only final joint error | 0.2301 rad |
| PD+GC final joint error | 0.1552 rad |
| Task-space mean position error | 0.0468 m |
| Task-space final position error | 0.0484 m |
| Task-space mean rotation error | 0.0206 rad |

### 解读

* **PD+GC 优于 PD-only**：在当前修正后的验证流程中，PD + Gravity Compensation 的 mean error 和 final error 均低于 PD-only，尤其 final error 差距较大（0.155 vs 0.230 rad），说明重力补偿有助于减小稳态跟踪误差。
* **Torque curve 对比**：PD+GC 的力矩范数明显低于 PD-only，表明重力补偿项承担了大部分静力学负担，PD 项只需处理偏差。
* **Task-space impedance**：位置误差约 4.7 cm、姿态误差约 0.02 rad，呈稳定周期性波动（匹配 0.2 Hz 目标轨迹），未出现发散。
* 以上结果均出自 MuJoCo 仿真，控制器增益并非最终优化值。

## Tests 与验证脚本

`tests/` 包含两类测试：

* **Automated tests**（默认）：不依赖 GUI、viewer 或摄像头的脚本，用 `def test_*()` 包装并由 pytest 自动发现。
* **Interactive / viewer validation**（标记为 `@pytest.mark.viewer`）：需要 MuJoCo viewer 显示和人工检查的脚本，默认跳过。

运行全部非 viewer 测试：

```bash
pytest -q -m "not viewer and not interactive"
```

运行全部测试（viewer 测试会因 `pytest.skip()` 跳过）：

```bash
pytest -q
```

详细说明见 `docs/testing.md`。

面试展示时建议优先使用 `推荐运行命令` 一节中的三条主线命令。

## 当前局限

当前项目的边界如下：

1. 不是完整 VLA 系统。
2. 不是完整人形机器人全身 Motion Retargeting 系统。
3. 不是完整足式机器人强化学习项目。
4. 控制器目前只在 MuJoCo 仿真中验证，尚未部署到真实机器人。
5. 当前 torque-level validation 仍然基于 position-actuated Panda MJCF，通过 neutralize_position_actuators 折中实现；更干净的后续工作是建立 torque-actuated Panda XML。
6. Hand retargeting 当前是手部输入到 Panda 末端目标的映射，不是人体全身动作到人形机器人的映射。
7. 单目深度方向映射和手掌姿态标定仍在调试。
8. retargeting 接口中已有 gripper-width command，但夹爪完整闭环接入仍需清理。

## 项目状态

已完成或部分完成：

* Franka Panda MuJoCo 模型加载；
* FK / IK / Jacobian / Pose IK demo；
* Null Space、Task Priority 和 Manipulability demo；
* 关节空间 PD 与 Gravity Compensation 对比实验流程；
* Task-space PD / Impedance-style control demo；
* MediaPipe hand input 封装；
* hand-to-Panda 末端目标映射原型。

面试前 TODO：

* optionally build a torque-actuated Panda XML for cleaner torque-control benchmarking；
* keep refining gains and task-space trajectory settings；
* 为 hand retargeting demo 补充短视频或 GIF；
* 区分真正自动化测试和交互式验证脚本；
* 清理或补全 `core/pose_ik.py` 等空/薄模块；
* 补充 retargeting 的相机标定和坐标系约定说明；
* 发布前清理旧运行日志，例如 `MUJOCO_LOG.TXT`。

