# 操作数据记录与策略验证接口

## 为何需要操作数据记录

在机器人系统开发中，操作数据（operation data）是连接底层控制与上层
策略（如模仿学习、强化学习、具身大模型）的桥梁。一份结构化的操作轨迹
记录可以让开发者：

- 验证动作映射管线的输出格式是否正确；
- 检查 workspace clamp、滤波等约束是否按预期工作；
- 为后续模仿学习 / 强化学习 / 具身模型策略验证提供标准化的输入格式。

当前模块的核心目标**不是**策略训练，而是定义数据接口和验证数据格式。

## 数据格式

所有数据通过 `core/operation_logger.py` 中的 `OperationLogger` 类记录。

### 字段说明

| 字段 | shape | 类型 | 说明 |
|---|---|---|---|
| step | — | int | 时间步序号 |
| time | — | float | 从起始开始的时间 [s] |
| target_pos | (3,) | float | 末端目标位置 [m] |
| current_pos | (3,) | float, optional | 末端实际位置 [m] |
| target_rot | (3, 3) or None | float | 末端目标姿态旋转矩阵 |
| current_qpos | (7,) or None | float | 当前关节角度 [rad] |
| action_pos_delta | (3,) or None | float | 当前步的位置增量 [m] |
| gripper_width | — | float | 夹爪宽度指令 [m] |
| valid | — | bool | 该步是否有效（手部检测成功等） |
| source | — | str | 数据来源标识（`mock_hand`, `mediapipe`, `sim_control` 等） |

### 支持的输出格式

- **CSV**（`save_csv`）：所有数值字段展开为扁平列，便于表格查看和简单脚本处理。
- **NPZ**（`save_npz`）：NumPy 压缩存档，保留数组形状，适合后续 Python 数据处理。

## 与模仿学习 / 强化学习 / 具身模型策略验证的关系

当前数据接口**不包含**：

- 策略训练算法；
- 奖励函数或值函数；
- 神经网络模型；
- 深度学习框架（PyTorch / TensorFlow）。

它仅提供以下能力：

- 将仿真中产生的操作指令与状态以结构化格式持久化；
- 使后续策略验证流程可以直接读取标准化的 `.csv` 或 `.npz` 文件；
- 为真机部署前的数据格式验证提供参考。

## 数据来源

当前数据来自 `scripts/generate_retargeting_demo.py` 生成的
**offline mock hand trajectory**，而非真实机器人或真实手部跟踪数据。

通过调用 `OperationLogger`，demo 脚本在 `results/retargeting/` 下额外输出：

- `mock_operation_trajectory.csv`
- `mock_operation_trajectory.npz`

## 后续扩展（真机部署规划）

如果后续接入真实机器人，需要补充：

- 真实机器人关节状态和时间戳同步；
- 安全检查和急停记录；
- 力 / 力矩传感器读数（可选）；
- 碰撞检测标志（可选）。
