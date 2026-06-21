# Robot Control & Hand-to-Panda Retargeting Framework

基于 MuJoCo 和 Franka Panda 的实时摄像头手部输入到机器人末端动作映射实验框架。

---

## 项目定位

本项目以 `demos/panda/demo_hand_retargeting_pd_gc.py` 为主实验入口，验证真实摄像头手部输入到
Panda 末端目标位姿的动作映射链路。项目重点包括：实时手部跟踪、手部到末端目标位姿映射、
MuJoCo 中的任务空间控制、CSV 数据记录、运行结果分析和参数对比。

**当前边界：**

- 不是完整人形机器人全身控制系统。
- 不是强化学习训练项目。
- 没有真机部署。
- 当前没有相机标定和米制深度映射，因此指标主要评估映射连续性、稳定性、可达性和
  MuJoCo 仿真跟踪响应，而不是绝对三维测量精度。

---

## 推荐运行流程

请在仓库根目录运行。

### 1. 运行主 demo（需摄像头）

```
python -m demos.panda.demo_hand_retargeting_pd_gc --camera-id 0 --duration 20 --no-camera-window
```

CSV 保存至 `results/hand_retargeting/runs/<run_id>/raw/hand_retargeting_run.csv`。

### 2. 分析单次运行

```
python scripts/analyze_hand_retargeting_run.py --input results/hand_retargeting/runs/<run_id>/raw/hand_retargeting_run.csv
```

在运行目录下生成分析图像和汇总指标。

### 3. 对比多次运行结果

```
python scripts/sweep_hand_retargeting_params.py --inputs \
  results/hand_retargeting/runs/run_a/raw/hand_retargeting_run.csv:default \
  results/hand_retargeting/runs/run_b/raw/hand_retargeting_run.csv:low_alpha
```

对比指标和图像输出至 `results/hand_retargeting/comparison/`。

### 4. 无摄像头控制器粗调

```
python scripts/run_preliminary_control_tuning.py
python scripts/plot_preliminary_control_tuning.py
```

使用固定正弦末端轨迹，对 5 组控制器参数进行无摄像头粗调。分数最低的参数组作为 demo 的初始参考值。
注意：粗调使用慢速正弦轨迹，所选参数增益偏低。真实手部运动速度更快，demo 默认使用更高增益，
以确保跟踪响应足够快。

### 5. 运行自动化测试

```
pytest -q -m "not viewer and not interactive"
```

---

## CSV 字段说明

由 `demo_hand_retargeting_pd_gc.py` 生成，被分析/对比脚本读取：

`timestamp`, `frame_id`, `detected_hand`, `detection_confidence` — 时间戳和检测状态

`wrist_x`, `wrist_y`, `wrist_z` — 手腕位置（图像坐标）

`pinch_ratio` — 捏合比（拇指尖与食指尖距离 / 手掌宽度）

`target_pos_x/y/z` — 目标末端位置；`filtered_target_pos_x/y/z` — 低通滤波后的目标位置

`target_quat_w/x/y/z` — 目标姿态四元数

`gripper_width` — 夹爪宽度；`workspace_clipped` — 是否触发工作空间限制

`actual_ee_pos_x/y/z` — 实际末端位置

`ee_position_error`, `ee_orientation_error` — 末端位置和姿态跟踪误差

`joint_q_1..7`, `joint_dq_1..7` — 7 个关节的位置和速度

`torque_norm`, `max_abs_torque` — 控制力矩范数和最大值

---

## 指标说明

| 指标 | 含义 | 单位 |
|------|------|------|
| detection_rate | 检测到手的帧数占比 | [0, 1] |
| mean_ee_position_error | 平均末端位置跟踪误差 | m |
| max_ee_position_error | 最大末端位置跟踪误差 | m |
| final_ee_position_error | 最后一帧末端位置误差 | m |
| target_position_smoothness | 目标位置相邻帧变化的平均范数 | m |
| target_position_jump_count | 目标位置突变次数（>5cm 跳变）| count |
| gripper_command_smoothness | 夹爪宽度相邻帧变化绝对值均值 | m |
| mean_torque_norm | 平均控制力矩范数 | Nm |
| max_torque_norm | 最大控制力矩范数 | Nm |
| workspace_clip_rate | 触发工作空间限制的帧占比 | [0, 1] |

---

## 项目结构

```
robot-control-retargeting-framework/
|-- demos/panda/
|   +-- demo_hand_retargeting_pd_gc.py    # 主实验入口：camera -> hand -> Panda -> CSV
|-- scripts/
|   +-- analyze_hand_retargeting_run.py    # 分析单次 demo CSV
|   +-- sweep_hand_retargeting_params.py  # 对比一份或多份 demo CSV
|   +-- run_preliminary_control_tuning.py # demo 前的控制器粗调
|   +-- plot_preliminary_control_tuning.py# 控制器粗调绘图
|-- core/
|   +-- dynamics_control.py                # Panda 力矩控制器
|   +-- pose_ik.py                        # 阻尼最小二乘位姿 IK
|-- retargeting/
|   +-- hand_to_panda.py                  # 手部 landmarks 到 Panda 目标位姿
|-- vision/
|   +-- hand_tracker.py                   # MediaPipe Hands 封装
|-- results/
|   +-- hand_retargeting/runs/            # 主 demo 运行数据
|   +-- preliminary_control/             # 控制器粗调结果
|-- docs/
|-- tests/
```

---

## 常见问题

- 主 demo 需要摄像头、OpenCV GUI 和 MuJoCo viewer。如果无摄像头可用，
  尝试 `--camera-id 0 --no-camera-window`。
- 控制器粗调（run_preliminary_control_tuning.py）不需要摄像头或 viewer。
- 标记为 `viewer` 或 `interactive` 的测试需要显示器；使用 `pytest -m "not viewer and not interactive"` 跳过。

---

## 当前局限

1. **非真机部署：** 所有实验仅在 MuJoCo 仿真中验证。
2. **非人体全身重定向：** 当前只做手到 Panda 末端，不涉及人体全身动作重定向。
3. **非强化学习项目：** 不包含强化学习训练环境或策略网络。
4. **无相机标定：** 深度映射依赖手掌尺寸估计，非米制，当前默认关闭。
5. **姿态映射为原型级别：** 大幅旋转时行为可能异常。
6. **摄像头结果依赖硬件：** 真实摄像头结果需要本地摄像头运行生成，无法在 CI 中自动复现。