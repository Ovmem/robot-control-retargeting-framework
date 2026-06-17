# 手部 Retargeting：坐标帧、映射流程与约束

## 概述

本文档描述当前 hand-to-Panda retargeting prototype 的工程设计边界。系统使用摄像头、MediaPipe Hands 和简化的映射方式，在 MuJoCo 仿真中控制 Franka Panda 末端目标位姿。

**范围：** 手 → Panda 末端目标（位置 + 姿态 + 夹爪）。非全身或 VR 动作重定向系统。

**当前状态：** 仿真原型。未部署到真实机器人。

---

## 1. 坐标帧

### A. 摄像头 / 图像帧

- **来源：** 单个 RGB 摄像头。
- **MediaPipe landmarks** 以归一化图像坐标返回：`x` ∈ [0, 1]（向右），`y` ∈ [0, 1]（向下），`z` ∈ [– 1, 1]（相对深度估计，非米制）。
- 演示中图像已经镜像（`mirror=True`），使手腕左右移动更自然。
- **无标定相机内参或外参。** MediaPipe 的 `z` 分量是每个关键点的相对深度，非米制距离。当前轴映射忽略图像 `z`（深度映射在 `map_position_from_image` 中被注释掉）。

### B. 手部局部帧

当世界坐标 landmarks（`landmarks_world`）可用时，构建右手正交帧：

```
wrist     = P[0]
index_mcp = P[5]
pinky_mcp = P[17]
mid_mcp   = P[9]

x_axis = normalize(index_mcp - pinky_mcp)
y_raw  = normalize(mid_mcp - wrist)
z_axis = normalize(cross(x_axis, y_raw))
y_axis = normalize(cross(z_axis, x_axis))
```

如果世界坐标不可用，则回退到图像坐标，但姿态估计质量会下降。

**注意：** 这是一个简化手部帧，不代表标定的人体手腕坐标帧。当手指弯曲或手部大幅旋转时，帧方向可能漂移。

### C. 机器人基坐标系

- Panda 机械臂在 MuJoCo 世界坐标系中运行。
- **固定起点**（`robot_origin`）定义在 home 末端位姿。
- 图像平面中的手部动作映射为相对该起点的笫卡尔增量：

  | 图像轴 | 机器人轴 | 缩放 |
  |---|---|---|
  | dx（左/右） | y（左/右） | `position_scale_xy` ≈ 2.2 |
  | dy（上/下） | z（上/下） | –0.8 × `position_scale_xy` |
  | 深度 | x（前/后） | 已注释 |

- 最终位置被钳位到以 `robot_origin` 为中心的安全工作空间内。

### D. 末端执行器帧

- 手部帧通过固定的旋转偏移与 Panda 末端执行器对齐：`R_robot_from_hand = R_euler([180°, 0°, 90°])`。
- 姿态经过低通滤波以减少抖动。
- **当前限制：** 姿态跟踪为原型级别；当手部大幅旋转时，映射行为可能异常。

## 2. 映射流程

```
第 1 步：捕获
    摄像头 → MediaPipe Hands → 21 个 landmarks（图像 + 世界）

第 2 步：归一化
    相对于手腕（P[0]）或手掌中心的 landmarks。

第 3 步：转换为机器人偏移
    图像 dx → 机器人 y（缩放 × 2.2）
    图像 dy → 机器人 z（缩放 × –1.76）
    手掌帧 → 通过 R_robot_from_hand 转换为目标旋转
    捏合比 → 夹爪宽度

第 4 步：钳位与滤波
    目标位置钳位到 Panda 工作空间内。
    对位置和旋转应用低通滤波。

第 5 步：发送到控制器
    target_pos, target_rot → 任务空间 PD / 阻抗式力矩控制器
```

## 3. 安全 / 约束

| 约束 | 状态 | 详情 |
|---|---|---|
| 工作空间限制 | ✅ 已实现 | `map_position_from_image` 中相对于 `robot_origin` 钳位 x/y/z |
| 低通滤波 | ✅ 已实现 | `LowPassFilter`（指数平滑），位置和旋转均适用。演示中 alpha ≈ 0.18 |
| 关节限制检查 | ❌ 规划中 | 仿真通过 MuJoCo 物理引擎遵守 Panda 关节限制；retargeter 中无独立检查 |
| 目标跳变抑制 | ❌ 规划中 | 无基于速度的 `target_pos` 增量限制。如果手部跟踪抖动，目标可能跳变 |
| 速度平滑 | ❌ 规划中 | 无明确的速度限制器；低通滤波提供轻微平滑 |
| 紧急停止 | ⚠️ Viewer 退出 | 演示在 MuJoCo viewer 关闭或 MediaPipe 窗口按下 ESC 时退出。无专用紧急停止按钮 |
| 真实机器人安全 | ❌ 不适用 | 仅限仿真。未实现真实机器人部署或安全层 |

## 4. 当前限制（总结）

- 非完整动作重定向系统（手 → 末端执行器）。
- 无标定 3D 摄像头或米制手部姿态。
- 姿态映射为原型级别；大幅旋转时可能行为异常。
- 深度（前后）映射当前已禁用。
- 无速度限制或目标跳变抑制。
- 夹爪指令已计算但未闭环控制。
- 仅单手控制（左手或右手，非双手）。
- 仅限仿真 — 未部署到真实机器人。

## 5. 离线 Mock Retargeting Demo

离线演示脚本位于 `scripts/generate_retargeting_demo.py`。

**功能：**

- 生成合成手部 landmarks 轨迹（手腕做利萨如运动，捏合比循环开闭）。
- 通过与摄像头演示相同的 `HandToPandaRetargeter` 管线处理该轨迹。
- 将 CSV 结果和 Matplotlib 图像保存到 `results/retargeting/`。

**验证内容：**

- 映射管线逻辑（位置缩放、轴分配、钳位）。
- 从捏合比到夹爪宽度的映射。
- 多秒轨迹上的低通滤波行为。

**不验证内容：**

- MediaPipe 跟踪质量或手部检测鲁棒性。
- 实时延迟。

运行：

```bash
python scripts/generate_retargeting_demo.py
```

无需摄像头、显示器或 MuJoCo viewer。

