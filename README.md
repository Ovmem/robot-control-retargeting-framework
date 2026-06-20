# Robot Control & Hand-to-Panda Retargeting Framework

基于 MuJoCo 和 Franka Panda 的真实摄像头手部输入到机器人末端动作映射实验框架。

---

## 项目定位

本项目基于 MuJoCo 和 Franka Panda，验证从真实摄像头手部输入到机器人末端目标位姿的动作映射链路。项目重点不只是可视化 demo，而是记录真实仿真运行数据，分析映射稳定性、末端跟踪误差、工作空间裁剪和夹爪指令响应，并基于指标调整映射参数。

**核心链路：**

`	ext
Camera
  -> MediaPipe hand landmarks
  -> Hand-to-Panda target pose mapping
  -> MuJoCo Panda task-space tracking
  -> Run data logging (CSV)
  -> Metrics and plots
  -> Parameter tuning
```

**当前边界：**

- 不是完整人形机器人全身控制系统。
- 不是强化学习训练项目。
- 没有真机部署。
- 由于尚未进行相机标定和真实 3D 手部位置标定，指标主要评估映射连续性、稳定性、可达性和 MuJoCo 仿真跟踪响应，而不是绝对三维测量精度。

---

## 推荐运行命令

请在仓库根目录运行。Windows PowerShell 下使用 python -m 执行 package-style demo。

### 1. 运行实时 hand retargeting demo 并保存真实运行数据

```bash
python -m demos.panda.demo_hand_retargeting_pd_gc --camera-id 0 --duration 20 --show-camera --output-dir results/hand_retargeting/runs
```

参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --camera-id | 0 | 摄像头编号 |
| --duration | 20 | 录制时长（秒） |
| --pos-scale | 2.2 | 手部运动到机器人运动的缩放 |
| --filter-alpha | 0.18 | 低通滤波系数（0=最平滑，1=无滤波） |
| --output-dir | results/hand_retargeting/runs | 输出根目录 |
| --show-camera | (不启用) | 显示摄像头监测窗口 |

Demo 自动创建以时间戳命名的实验目录：

`	ext
results/hand_retargeting/runs/20260620_143022/
+-- raw/
|   +-- hand_retargeting_run.csv
+-- figures/          # 运行后通过分析脚本生成
+-- metrics/          # 运行后通过分析脚本生成
```

CSV 记录字段：timestamp, frame_id, detected_hand, detection_confidence, wrist position, pinch_ratio, target position (raw + filtered), target quaternion, gripper_width, workspace_clipped, actual end-effector position, ee_position_error, ee_orientation_error, joint positions/velocities, torque_norm, max_abs_torque。

### 2. 分析一次 demo 运行

```bash
python scripts/analyze_hand_retargeting_run.py --input results/hand_retargeting/runs/<run_id>/raw/hand_retargeting_run.csv
```

生成的图：

| 输出 | 内容 |
|------|------|
| figures/target_vs_actual_ee_position.png | 目标 vs 实际末端位置 |
| figures/ee_position_error.png | 末端位置跟踪误差 |
| figures/gripper_width_and_pinch.png | 夹爪宽度和捏合比 |
| figures/target_smoothness.png | 目标位置平滑性 |
| figures/detection_status.png | 手部检测状态 |
| figures/workspace_clipping.png | 工作空间裁剪事件 |
| figures/torque_norm.png | 控制力矩 |

汇总指标保存到 metrics/hand_retargeting_metrics.csv。

### 3. 参数扫描（基于同一段真实数据）

```bash
python scripts/sweep_hand_retargeting_params.py --input results/hand_retargeting/runs/<run_id>/raw/hand_retargeting_run.csv
```

不重新打开摄像头，而是读取已录制的手部数据，用不同映射参数重新计算 target pose 并比较指标。

扫描参数：position_scale (2.2, 3.5), smoothing_alpha (0.08, 0.18, 0.40), workspace_bounds, orientation_mapping on/off, gripper_mapping on/off。

输出：param_sweep/raw/results.csv, param_sweep/figures/*.png, param_sweep/metrics/best_params.csv。

### 4. 预实验：任务空间控制器参数粗调

```bash
python scripts/run_preliminary_control_tuning.py
python scripts/plot_preliminary_control_tuning.py
```

使用固定的末端目标轨迹（正弦运动），测试 5 组任务空间控制参数（soft / balanced / responsive / aggressive / torque_limited），选择综合评分最低的作为推荐参数。不依赖摄像头，仅用于确保 hand retargeting demo 开始前控制器参数基本合理。

运行后查看推荐参数：
```bash
cat results/preliminary_control/metrics/best_control_params.csv
```

推荐参数可通过 --kp-pos, --kd-pos, --kp-ori, --kd-ori, --torque-limit, --max-target-step 传入真实 demo。

| 模式 | kp_pos | kd_pos | kp_ori | kd_ori | torque_limit | Mean EE err [m] | Mean torque [Nm] | Score |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| soft | 80 | 16 | 20 | 4 | 40 | 0.051 | 23.0 | 0.781 |
| balanced | 120 | 24 | 35 | 7 | 55 | 0.051 | 23.1 | 0.780 |
| responsive | 160 | 32 | 45 | 9 | 70 | 0.051 | 23.3 | 0.781 |
| aggressive | 220 | 40 | 60 | 12 | 90 | 0.051 | 23.7 | 0.782 |
| torque_limited | 140 | 28 | 40 | 8 | 35 | 0.051 | 23.2 | 0.781 |

### 5. 自动化测试

```bash
pytest -q -m "not viewer and not interactive"
```

---

## 指标说明

| 指标 | 含义 | 单位 |
|------|------|------|
| detection_rate | 检测到手的帧数占比 | [0, 1] |
| mean_ee_position_error | 平均末端位置跟踪误差 | m |
| max_ee_position_error | 最大末端位置跟踪误差 | m |
| target_position_smoothness | 目标位置相邻帧变化的平均范数 | m |
| target_position_jump_count | 目标位置突变次数（>5cm 跳变） | count |
| workspace_clip_rate | 触发工作空间限制的帧占比 | [0, 1] |
| gripper_command_smoothness | 夹爪宽度相邻帧变化绝对值均值 | m |
| mean_torque_norm | 平均控制力矩范数 | Nm |
| max_torque_norm | 最大控制力矩范数 | Nm |

---

## 项目结构

`	ext
robot_control_retargeting_framework/
+-- demos/panda/
|   +-- demo_hand_retargeting_pd_gc.py   # 主实验：实时 hand retargeting + 数据采集
|
+-- scripts/
|   +-- analyze_hand_retargeting_run.py      # 运行数据分析 + 绘图
|   +-- sweep_hand_retargeting_params.py     # 参数扫描（基于同一段真实数据）
|   +-- run_preliminary_control_tuning.py    # 预实验：粗调控制器参数
|   +-- plot_preliminary_control_tuning.py   # 预实验绘图
|
+-- core/
|   +-- dynamics_control.py                 # Panda 力矩控制器
|   +-- pose_ik.py                          # 阻尼最小二乘位姿 IK
|
+-- retargeting/
|   +-- hand_to_panda.py                    # 手部 landmarks 到 Panda 目标位姿
|
+-- vision/
|   +-- hand_tracker.py                     # MediaPipe Hands 封装（不自开窗口）
|
+-- results/
|   +-- hand_retargeting/runs/              # 主实验运行数据
|   +-- preliminary_control/                # 预实验控制器调参
|
+-- tests/                                  # pytest 自动化测试
+-- docs/
|   +-- retargeting.md                      # 动作映射坐标系文档
|   +-- testing.md                          # 测试策略说明
+-- README.md
```

---

## 面试展示流程

`	ext
Step 1: Run camera-based demo and show the MuJoCo response.
    python -m demos.panda.demo_hand_retargeting_pd_gc --camera-id 0 --duration 20 --show-camera

Step 2: Open generated plots to explain mapping stability and tracking error.
    python scripts/analyze_hand_retargeting_run.py --input results/hand_retargeting/runs/<id>/raw/hand_retargeting_run.csv

Step 3: Show parameter sweep results, explain how smoothing, scale, and workspace bounds affect the mapping.
    python scripts/sweep_hand_retargeting_params.py --input results/hand_retargeting/runs/<id>/raw/hand_retargeting_run.csv
```

---

## 环境配置

```bash
pip install -r requirements.txt
```

主要依赖：mujoco, numpy, scipy, matplotlib, opencv-python, mediapipe, pytest。

---

## 当前局限

1. 非真机部署：所有实验仅在 MuJoCo 仿真中验证。
2. 非人形全身重定向：当前只做手到 Panda 末端，不涉及人体全身动作重定向。
3. 非 RL 项目：不包含强化学习训练环境或策略网络。
4. 位置 actuator 折中：Panda MJCF 使用 affine-bias position actuators，当前通过 neutralize_position_actuators 折中实现力矩控制。
5. 无相机标定：深度映射依赖手掌尺寸估计，非米制，当前默认关闭。
6. 姿态映射为原型级别：大幅旋转时行为可能异常。
7. 姿态误差指标带宽受限：orientation error 记录在 CSV 中，但准确度受 retargeter 姿态估计精度影响。
