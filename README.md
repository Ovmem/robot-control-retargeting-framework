# Robot Control & Hand-to-Panda Retargeting Framework

基于 MuJoCo 和 Franka Panda 的机器人运动控制与手部动作映射原型。

---

## 项目定位

本项目面向**机器人运动控制 / 仿真算法 / Motion Retargeting 面试**展示，核心是在 MuJoCo 仿真环境中驱动 Franka Panda 七自由度机械臂，并通过 MediaPipe 手部输入到末端目标位姿的动作映射链路完成实验对比。

**主线能力：**

- Franka Panda 七自由度机械臂的运动学（FK / IK / Jacobian / Pose IK）与冗余控制（Null Space、Task Priority、Manipulability）demo；
- 关节空间 PD / PD + 重力补偿的力矩控制 sanity check；
- **手部关键点 → Panda 末端位置、姿态、夹爪宽度**的动作映射消融分析（主实验）；
- 控制实验日志、曲线绘制与可复现消融分析流程。

**明确边界：**

- 不是完整人形机器人全身 Motion Retargeting 系统；
- 不是完整强化学习（RL）训练项目；
- 不是真机部署方案；
- 不是 VLA（Vision-Language-Action）或具身大模型系统。

---

## 项目亮点

- **Panda 七自由度机械臂 MuJoCo 仿真**：完整 Panda MJCF 模型，支持力矩级控制。
- **手部动作映射消融分析**：同一段手部输入，对比 full pipeline 与去掉滤波、工作空间限制、速度限制、丢帧保持、姿态映射、夹爪映射后的差异。
- **手部关键点录制**：支持 webcam 录制真实手部序列。
- **离线验证**：无需摄像头即可复现动作映射消融。
- **Panda 关节空间控制 sanity check**：PD、PD+GC、低增益、力矩限制的 baseline 对比。
- **基础学习 demo 归档**：简化机械臂 FK/IK/Jacobian demo 已标为基础学习，不干扰主线。

---

## 实验：手部动作映射消融分析（主实验）

这是项目的核心实验。固定同一段手部关键点输入，系统性地移除映射管道中的模块，观察目标平滑性、末端跟踪误差和力矩的变化。

### 消融组

| 模式 | 说明 |
|------|------|
| full_pipeline | 完整链路：工作空间限制 + 低通滤波 + 目标速度限制 + 丢帧保持 + 姿态映射 + 捏合映射 |
| no_smoothing | 去除低通滤波 |
| no_workspace_clamp | 去除工作空间限制 |
| no_rate_limit | 去除目标速度限制 |
| no_dropout_hold | 检测不到手时不保持上一帧目标 |
| no_orientation_mapping | 不使用手掌姿态映射（固定目标姿态） |
| no_pinch_gripper | 夹爪宽度固定 |

### 当前结果（mock 手部轨迹，6s @ 30fps，MuJoCo 响应仿真）

| 模式 | Mean EE err [m] | Max EE err [m] | Target jerk RMS [m/s^3] | RMS torque [Nm] | Workspace clip ratio |
|------|:---:|:---:|:---:|:---:|:---:|
| Full pipeline | 0.102 | 0.188 | 130.4 | 66.1 | 0.00 |
| No smoothing | 0.102 | 0.188 | 135.0 | 66.1 | 0.00 |
| No workspace clamp | 0.113 | 0.232 | 107.8 | 72.0 | 0.00 |
| No rate limit | 0.117 | 0.206 | 106.4 | 75.8 | 0.00 |
| No dropout hold | 0.102 | 0.188 | 2395.3 | 66.3 | 0.00 |
| No orientation | 0.091 | 0.192 | 130.4 | 65.9 | 0.00 |
| No pinch gripper | 0.102 | 0.188 | 130.4 | 66.1 | 0.00 |

> 注意：
> - 以上结果基于 MuJoCo 仿真和 mock 手部轨迹，不构成真机实验结论。
> - **Workspace clip ratio** 在当前 mock 轨迹下为 0（轨迹未越界）；在更大幅度的手部运动中该指标会生效。
> - **Target jerk RMS** 以 m/s^3 计量，第三阶导数数值较大属正常（no_dropout_hold 因连续丢帧导致目标跳变，jerk 剧增）。
> - **No orientation mapping** 使得位置跟踪误差略降低（控制器不需要同时补偿姿态变化），但这不代表姿态映射本身有缺陷。
> - 姿态误差指标当前为占位（NaN），需进一步接入完整的旋转矩阵跟踪评估。

### 初步解读

- **Filter（no_smoothing）**：开关对稳态误差影响不大，但影响目标平滑性（jerk 略增）。
- **Workspace clamp（no_workspace_clamp）**：关闭后目标可超出仿真安全范围，末端误差和力矩均上升。
- **Rate limit（no_rate_limit）**：关闭速度限制后，轨迹瞬时速度和力矩峰值升高，torque smoothness 变差。
- **Dropout hold（no_dropout_hold）**：间歇性丢帧导致目标反复跳变到原点，jerk 大幅飙升（130 → 2395），是影响平滑性的关键模块。
- **Orientation mapping（no_orientation_mapping）**：固定姿态使位置跟踪更专注，但代价是丧失末端姿态控制能力。
- **Pinch gripper（no_pinch_gripper）**：对位置和力矩影响最小，只影响夹爪指令。

---

## 实验：Panda 关节空间控制 Sanity Check

本实验作为底层控制器 baseline 验证，不直接用于 hand retargeting 评估。使用确定性目标轨迹（阶跃 + 正弦），对比关节空间 PD 的各变体。

### 组别（仅保留关节空间可比较的模式）

| 模式 | 控制类型 | 增益 | 重力补偿 | 力矩限制 |
|------|---------|------|---------|---------|
| pd_only | Joint PD | 默认 | 关闭 | 默认 |
| pd_gc | Joint PD | 默认 | 开启 | 默认 |
| pd_gc_low_gain | Joint PD | 50% | 开启 | 默认 |
| pd_gc_torque_clipped | Joint PD | 默认 | 开启 | 更严格 |

### 当前结果（MuJoCo 仿真，5 秒/组）

| 模式 | Mean joint err [rad] | Final joint err [rad] | RMS torque [Nm] | Max torque [Nm] | Torque smoothness |
|------|:---:|:---:|:---:|:---:|:---:|
| PD only | 0.207 | 0.308 | 16.6 | 22.0 | 0.031 |
| PD + GC | 0.204 | 0.304 | 26.7 | 29.5 | 0.028 |
| PD + GC (low gain) | 0.214 | 0.335 | 23.6 | 24.5 | 0.014 |
| PD + GC (clipped) | 0.204 | 0.304 | 26.7 | 29.5 | 0.028 |

> **Task-space PD** 和 **Computed torque** 模式已在脚本中实现，不在此表中列出（它们分别跟踪末端位姿和使用逆动力学前馈，不宜与 joint-space 指标直接混比）。可运行 `python scripts/run_panda_control_ablation.py` 查看完整结果。

---

## 手部动作映射链路

```text
摄像头 / Mock hand trajectory
  → MediaPipe Hands 21 landmarks
  → wrist / palm frame / pinch ratio
  → Panda 末端目标位置（y/z 平面，深度映射默认关闭）
  → Panda 末端目标姿态（旋转矩阵，固定偏移对齐）
  → 夹爪宽度指令（pinch ratio → gripper width）
  → Task-space PD / Impedance 控制器
  → MuJoCo Panda 响应
```

**当前映射约束：**

- 图像 x（左右）→ 机器人 y，图像 y（上下）→ 机器人 z，缩放系数可调（默认 ~2.2）。
- 深度方向映射（z → x）为可选原型，默认关闭。
- 姿态映射基于 MediaPipe world landmarks 构建手掌坐标系，通过固定旋转偏移与 Panda 末端对齐。
- 夹爪宽度通过拇指-食指 pinch ratio 映射到 [0, 0.04] m。
- 目标经过 workspace clamp、速度限制（rate limit）和低通滤波。

详细坐标帧约定见 `docs/retargeting.md`。当前为单目 RGB + MediaPipe 原型，无真实相机内外参标定。

---

## 推荐运行命令

请在仓库根目录运行。Windows PowerShell 下使用 `python -m` 执行 package-style demo。

### 手部动作映射消融分析（主实验）

```bash
# 1. 录制真实手部序列（需摄像头）
python scripts/record_hand_sequence.py --camera-id 0 --duration 10

# 2. 运行手部动作映射消融实验
python scripts/run_hand_retargeting_ablation.py --input results/retargeting/camera/raw/hand_sequence.csv

# 3. 绘制结果
python scripts/plot_hand_retargeting_ablation.py

# 无摄像头环境下使用 mock 输入：
python scripts/run_hand_retargeting_ablation.py
python scripts/plot_hand_retargeting_ablation.py
```

### Panda 控制器 Sanity Check

```bash
# 关节空间 PD vs PD+GC 对比
python -m demos.panda.demo_joint_pd_gc

# 任务空间阻抗控制 demo
python -m demos.panda.demo_task_space_impedance

# 控制消融分析（4 组联合空间模式 + 2 组实验模式）
python scripts/run_panda_control_ablation.py
python scripts/plot_panda_control_ablation.py
```

### 其他

```bash
# 离线 retargeting 验证
python scripts/generate_retargeting_demo.py

# 实时手部 Retargeting（需摄像头 + 显示器）
python -m demos.panda.demo_hand_retargeting_pd_gc --duration 10

# 自动化测试
pytest -q -m "not viewer and not interactive"
```

---

## 项目结构

```text
robot_control_retargeting_framework/
├── core/                          # 核心控制与运动学模块
│   ├── dynamics_control.py        # Panda 力矩控制器（PD, GC, task-space, computed torque）
│   ├── pose_ik.py                 # 阻尼最小二乘位姿 IK
│   └── operation_logger.py        # 操作数据记录接口
│
├── retargeting/
│   └── hand_to_panda.py           # 手部 landmarks → Panda 末端目标位姿 / 夹爪宽度
│
├── vision/
│   ├── hand_tracker.py            # MediaPipe Hands 封装
│   └── mediapipe_hand.py          # 早期独立 MediaPipe demo
│
├── demos/
│   ├── panda/                     # [主线] Panda 七自由度机械臂控制 demo
│   ├── ik/                          # 简化机械臂 IK demo（2D, 3-DoF, 3D）
│   ├── jacobian/                     # Jacobian 速度映射 demo
│   ├── nullspace/                    # 简化 Null Space 控制 demo
│   ├── orientation/                  # 姿态误差与四元数 demo
│   ├── pose_ik/                      # 6D 位姿 IK demo
│   ├── trajectory/                   # 轨迹跟踪 demo
│   ├── control/                      # 简化控制 demo
│   ├── retargeting/                  # 早期 retargeting 探索 demo
│   ├── visualization/                # 模型可视化辅助脚本
│   ├── kinematics/                   # 运动学相关 demo
│   └── redundancy/                   # 冗余度分析 demo
│
├── scripts/
│   ├── record_hand_sequence.py             # 录制手部关键点序列
│   ├── run_hand_retargeting_ablation.py    # [主实验] 手部动作映射消融分析
│   ├── plot_hand_retargeting_ablation.py   # 消融结果绘图
│   ├── run_panda_control_ablation.py       # [Sanity] 关节空间控制消融
│   ├── plot_panda_control_ablation.py      # 控制消融绘图
│   ├── plot_dynamics_results.py            # 旧版控制结果绘图（兼容）
│   └── generate_retargeting_demo.py        # 离线 Mock Retargeting
│
├── models/
│   └── panda/                     # Franka Panda MJCF 模型
│
├── results/
│   ├── panda_control/             # Panda 关节控制 sanity check
│   │   ├── raw/figures/metrics/
│   ├── retargeting/               # 动作映射结果
│   │   ├── ablation/raw/metrics/figures/
│   │   └── camera/raw/
│   ├── dynamics/                  # 旧版控制结果（兼容路径）
│   └── legacy/                    # 早期基础学习 demo 结果图归档
│
├── tests/                         # pytest 自动化测试
├── docs/
│   ├── retargeting.md             # 手部动作映射坐标系与约束文档
│   ├── testing.md                 # 测试策略说明
│   └── data_logging.md            # 操作数据记录接口说明
└── README.md
```

> 注：`results/*/raw/` 目录下的 CSV 文件通过运行对应脚本自动生成。

---

## 当前局限

1. **非真机部署**：所有控制器仅在 MuJoCo 仿真中验证。
2. **非人形全身重定向**：当前只做手 → Panda 末端，不涉及人体全身动作重定向。
3. **非 RL 项目**：不包含强化学习训练环境或策略网络。
4. **位置 actuator 折中**：Panda MJCF 使用 affine-bias position actuators，当前通过 `neutralize_position_actuators` 折中实现力矩级控制。
5. **姿态映射为原型级别**：无真实手腕姿态标定，大幅旋转时行为可能异常。
6. **深度映射默认关闭**：依赖手掌尺寸估计深度，非米制。
7. **夹爪无闭环控制**：gripper width 指令已输出至 ctrl[7]，但未做力或位置闭环。
8. **姿态误差指标当前为占位**：MuJoCo 响应仿真中 orientation error 尚未接入完整计算。

---

## 面试展示建议

一键复现所有实验：

```bash
# 核心实验：手部动作映射消融
python scripts/run_hand_retargeting_ablation.py
python scripts/plot_hand_retargeting_ablation.py

# 控制器 baseline
python scripts/run_panda_control_ablation.py
python scripts/plot_panda_control_ablation.py

# 离线 retargeting
python scripts/generate_retargeting_demo.py
```

建议面试时先展示手部动作映射消融曲线（`target_position_ablation.png`、`summary_metrics_ablation.png`），再展示控制器 baseline，最后用 `demo_hand_retargeting_pd_gc` 做实时交互或展示离线 retargeting 曲线。

---

## 环境配置

```bash
pip install -r requirements.txt
```

主要依赖：`mujoco`, `numpy`, `scipy`, `matplotlib`, `opencv-python`, `mediapipe`, `pytest`。

推荐 Python 3.10+。请在仓库根目录运行命令。

---

## Future Work

- Torque-actuated Panda XML：消除 position actuator 折中。
- 控制器增益系统整定。
- 姿态误差指标接入完整评估。
- 相机标定与米制深度映射。
- 夹爪闭环控制。
- 手部 demo 实时录屏。
