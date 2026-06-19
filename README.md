# Robot Control & Hand-to-Panda Retargeting Framework

基于 MuJoCo 和 Franka Panda 的机器人运动控制与手部动作映射原型。

---

## 项目定位

本项目面向**机器人运动控制 / 仿真算法 / Motion Retargeting 实习或岗位面试**展示，核心是在 MuJoCo 仿真环境中驱动 Franka Panda 七自由度机械臂，并通过 MediaPipe 手部输入到末端目标位姿的动作映射链路完成交互式演示。

**主线能力：**

- Franka Panda 七自由度机械臂的运动学（FK / IK / Jacobian / Pose IK）与冗余控制（Null Space、Task Priority、Manipulability）demo；
- 关节空间 PD、PD + 重力补偿（Gravity Compensation）、任务空间 PD / 阻抗控制的力矩控制对比与消融分析；
- MediaPipe 手部关键点 → Panda 末端位置、姿态、夹爪宽度的动作映射原型；
- 控制实验日志、曲线绘制与可复现消融分析流程。

**明确边界：**

- 不是完整人形机器人全身 Motion Retargeting 系统；
- 不是完整强化学习（RL）训练项目；
- 不是真机部署方案；
- 不是 VLA（Vision-Language-Action）或具身大模型系统。

---

## 项目亮点

- **Panda 七自由度机械臂 MuJoCo 仿真**：完整的 Panda MJCF 模型（含夹爪），支持力矩级控制与位置级 actuator 兼容折中。
- **关节空间 PD / PD + GC / 计算力矩 / 任务空间控制**：多种控制器统一接口，支持力矩裁剪和零空间投影。
- **手部动作映射原型**：MediaPipe 21 个手部关键点 → 末端位置（y/z）、目标姿态（旋转矩阵）、夹爪宽度指令（pinch ratio 映射）。
- **控制消融分析**：6 组对比实验，记录 mean / final / max 关节误差、RMS / max / smoothness 力矩指标，一键运行。
- **离线验证**：无需摄像头、无需 GUI 即可验证动作映射管线的逻辑。
- **基础学习 demo 归档**：早期简化机械臂的 FK / IK / Jacobian / Null Space 学习 demo 已标记为 fundamentals，不干扰主线。

---

## 推荐运行命令

请在仓库根目录运行。Windows PowerShell 下使用 `python -m` 执行 package-style demo。

### Panda 控制主线

```bash
# 关节空间 PD vs PD + 重力补偿对比
python -m demos.panda.demo_joint_pd_gc

# 任务空间阻抗控制 demo
python -m demos.panda.demo_task_space_impedance

# 绘制已有实验数据曲线
python scripts/plot_dynamics_results.py
```

### 消融分析（无 GUI，2-5 秒可完成）

```bash
# 运行全部消融组别（默认 5 秒/组，可用 --duration 调整）
python scripts/run_panda_control_ablation.py

# 绘制消融结果曲线与指标表
python scripts/plot_panda_control_ablation.py
```

### 离线 retargeting 验证（无摄像头）

```bash
python scripts/generate_retargeting_demo.py
```

### 实时手部 Retargeting（需摄像头）

```bash
python -m demos.panda.demo_hand_retargeting_pd_gc --duration 10
```

### 自动化测试

```bash
pytest -q -m "not viewer and not interactive"
```

> 注意：viewer / webcam 相关测试默认跳过。需要 MuJoCo viewer 或摄像头的脚本请手动运行检查。

---

## 项目结构

```text
robot_control_retargeting_framework/
├── core/                          # 核心控制与运动学模块
│   ├── dynamics_control.py        # Panda 力矩控制器（PD, GC, task-space, computed torque）
│   ├── pose_ik.py                 # 阻尼最小二乘位姿 IK
│   ├── jacobian_3d.py             # 简化 3D 机械臂数值雅可比
│   ├── kinematics.py / kinematics_3d.py / controller.py
│   └── operation_logger.py        # 操作数据记录接口
│
├── retargeting/
│   └── hand_to_panda.py           # 手部 landmarks → Panda 末端目标位姿 / 夹爪宽度
│
├── vision/
│   ├── hand_tracker.py            # 可复用 MediaPipe Hands 封装
│   └── mediapipe_hand.py          # 早期独立 MediaPipe demo
│
├── demos/
│   ├── panda/                     # [面试展示主线] Panda 七自由度机械臂控制 demo
│   │   ├── demo_joint_pd_gc.py
│   │   ├── demo_task_space_impedance.py
│   │   ├── demo_hand_retargeting_pd_gc.py
│   │   ├── demo_joint_limit_avoidance.py
│   │   ├── demo_manipulability_nullspace.py
│   │   ├── demo_pose_task_priority_ik.py
│   │   ├── demo_task_priority_ik.py
│   │   └── demo_nullspace_motion.py
│   └── fundamentals/              # [基础学习 demo] 简化机械臂 FK/IK/Jacobian/Null Space
│       ├── ik/                    #   IK demo（2D, 3-DoF, 3D）
│       ├── jacobian/              #   Jacobian 速度映射 demo
│       ├── nullspace/             #   简化 Null Space 控制 demo
│       ├── orientation/           #   姿态误差与四元数 demo
│       ├── pose_ik/               #   6D 位姿 IK demo
│       ├── trajectory/            #   轨迹跟踪 demo
│       ├── control/               #   简化控制 demo
│       ├── retargeting/           #   早期 retargeting 探索 demo
│       └── visualization/         #   模型可视化辅助脚本
│
├── models/
│   ├── panda/                     # Franka Panda MJCF 模型和 mesh 资源
│   └── arm2d.xml, arm3d.xml ...   # 基础学习模型（详见 demos/fundamentals/）
│
├── scripts/
│   ├── run_panda_control_ablation.py      # 控制消融分析（6 组对比）
│   ├── plot_panda_control_ablation.py     # 消融结果绘图
│   ├── plot_dynamics_results.py           # 旧版控制结果绘图（兼容路径）
│   └── generate_retargeting_demo.py       # 离线 Mock Retargeting
│
├── results/
│   ├── panda_control/             # [主结果] Panda 控制消融分析
│   │   ├── raw/                   #   各模式 CSV 日志
│   │   ├── figures/               #   误差曲线 / 力矩曲线 / 指标柱状图
│   │   └── metrics/               #   汇总指标表
│   ├── retargeting/               # 动作映射结果（离线验证）
│   ├── dynamics/                  # 旧版控制结果（兼容路径）
│   └── legacy/                    # 早期基础学习 demo 结果图归档
│
├── tests/                         # pytest 测试（non-viewer / non-interactive 为自动化测试）
├── docs/
│   ├── retargeting.md             # 手部动作映射坐标帧与约束文档
│   ├── testing.md                 # 测试策略说明
│   └── data_logging.md            # 操作数据记录接口说明
├── requirements.txt
└── README.md
```

---

## 控制消融分析

### 实验设计

`scripts/run_panda_control_ablation.py` 运行 6 组控制对比实验，使用相同的确定性目标轨迹（初始保持 → Hermite 平滑过渡 → 正弦调制保持）:

| 模式 | 控制类型 | 增益 | 重力补偿 | 力矩限制 |
|------|---------|------|---------|---------|
| pd_only | Joint PD | 默认 | 关闭 | 默认 |
| pd_gc | Joint PD | 默认 | 开启 | 默认 |
| pd_gc_low_gain | Joint PD | 50% | 开启 | 默认 |
| pd_gc_torque_clipped | Joint PD | 默认 | 开启 | 更严格 |
| computed_torque | ID + PD | 默认 | 隐含 | 默认 |
| task_space_pd_gc | Task PD | 默认 | 开启 | 默认 |

### 当前结果（MuJoCo 仿真，5 秒/组）

| 模式 | Mean joint err [rad] | Final joint err [rad] | RMS torque [Nm] | Max torque [Nm] | Torque smoothness |
|------|:---:|:---:|:---:|:---:|:---:|
| PD only | 0.207 | 0.308 | 16.6 | 22.0 | 0.031 |
| PD + GC | 0.204 | 0.304 | 26.7 | 29.5 | 0.028 |
| PD + GC (low gain) | 0.214 | 0.335 | 23.6 | 24.5 | 0.014 |
| PD + GC (clipped) | 0.204 | 0.304 | 26.7 | 29.5 | 0.028 |
| Computed torque | 0.202 | 0.299 | 33.0 | 40.9 | 0.039 |
| Task-space PD + GC | - | - | 22.1 | 22.7 | 0.007 |

> 注意：
> - 以上结果出自 MuJoCo 仿真，控制器增益并非最终优化值，不构成真机实验结论。
> - 「Task-space PD + GC」一行 joint-level 误差不适用（该控制器跟踪末端位姿而非关节角），仅列示力矩指标。
> - `pd_gc_torque_clipped` 在当前轨迹下力矩未触及更严格的限制边界，因此指标与 `pd_gc` 一致；在更大步长或更高增益下约束会生效。
> - Torque smoothness 定义为相邻控制步力矩差分矢量的平均范数，值越小表示力矩越平滑。

### 初步解读

- **PD + GC 优于 PD only**：重力补偿减小了稳态跟踪误差（受 position actuator 折中影响，差距不大），同时力矩范数更高（GC 项承担了重力负载）。
- **低增益 PD + GC**：误差略有上升，力矩更平滑（smoothness 从 0.028 降至 0.014），体现了 gain 与 smoothness 的 trade-off。
- **Computed torque**：通过逆动力学前馈误差补偿，mean / final error 最低，但力矩变化更剧烈（smoothness 0.039）。
- **Task-space PD + GC**：力矩分布较集中（RMS 22.1 / max 22.7 Nm），smoothness 最好（0.007），但 joint-level 误差需转为 task-space 评估。

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
- 目标经过 workspace clamp 和低通滤波（指数平滑）以减少抖动。

详细坐标帧约定见 `docs/retargeting.md`。当前为单目 RGB + MediaPipe 原型，无真实相机内外参标定。

---

## 当前局限

1. **非真机部署**：所有控制器仅在 MuJoCo 仿真中验证。
2. **非人形全身重定向**：当前只做手 → Panda 末端，不涉及人体全身动作重定向。
3. **非 RL 项目**：不包含强化学习训练环境或策略网络。
4. **位置 actuator 折中**：Panda MJCF 默认使用 affine-bias position actuators，当前通过 `neutralize_position_actuators`（ctrl = qpos）折中实现力矩级控制；更干净的方案是建立 torque-actuated Panda XML（future work）。
5. **姿态映射为原型级别**：手掌坐标系无真实手腕姿态标定，大幅旋转时行为可能异常。
6. **深度映射依赖手掌尺寸**：非米制深度，当前默认关闭。
7. **夹爪无闭环控制**：gripper width 指令已输出至 ctrl[7]，但未做力或位置闭环。
8. **无速度限制或目标跳变抑制**：低通滤波提供轻度平滑，但无专用限速器。
9. **控制器增益非最优**：当前增益为初步整定值，消融分析展示的是对比流程而非优化 benchmark。

---

## 面试展示建议

一键复现所有实验结果：

```bash
# 1. Panda 关节空间控制 demo
python -m demos.panda.demo_joint_pd_gc

# 2. 消融分析（无 GUI）
python scripts/run_panda_control_ablation.py
python scripts/plot_panda_control_ablation.py
# 结果见 results/panda_control/figures/

# 3. 离线 retargeting 验证（无摄像头）
python scripts/generate_retargeting_demo.py
# 结果见 results/retargeting/
```

建议面试时先展示 `demo_joint_pd_gc` 的控制流程，再展示消融曲线图（`control_ablation_error.png`、`control_ablation_summary.png`），最后用 `demo_hand_retargeting_pd_gc` 做实时交互或展示离线 retargeting 曲线。

---

## 环境配置

```bash
pip install -r requirements.txt
```

主要依赖：`mujoco`, `numpy`, `scipy`, `matplotlib`, `opencv-python`, `mediapipe`, `pytest`。

推荐 Python 3.10 或 3.11。请在仓库根目录运行命令。

---

## Future Work

- Torque-actuated Panda XML：构建纯力矩驱动的 Panda 模型，消除 position actuator 折中。
- 控制器增益系统整定：通过更系统的搜索方法优化各模式增益。
- 手部 demo 实时录屏：当前离线曲线用于无摄像头验证，实时 demo 可配合录屏展示。
- 相机标定与深度映射：接入标定后的 RGB-D 相机或双目视觉以获取米制深度。
- 夹爪闭环控制：接入力或位置反馈。
