# Robot Control & Hand Retargeting Framework

基于 MuJoCo 的机器人运动控制与手部动作映射验证框架。项目以 Franka Panda 七自由度机械臂为主要对象，围绕运动学、逆运动学、冗余控制、动力学控制和视觉手部输入映射进行算法验证，目标是构建一个面向机器人运控、遥操作和动作映射算法开发的可扩展仿真平台。

当前项目已完成 Franka Panda 机械臂的正运动学、雅可比矩阵、位置逆运动学、六维位姿逆运动学、零空间控制、任务优先级控制、可操作度优化、PD 控制、重力补偿和任务空间控制等模块，并初步跑通基于 MediaPipe Hands 的手部输入到机械臂末端目标位姿的实时闭环验证。

## 1. 项目简介

本项目主要面向以下问题：

* 如何在 MuJoCo 中搭建七自由度机械臂控制仿真环境；
* 如何实现机械臂正运动学、雅可比矩阵和逆运动学求解；
* 如何利用冗余自由度完成关节中心优化和可操作度优化；
* 如何从关节空间 PD 控制扩展到 PD + 重力补偿和任务空间力矩控制；
* 如何将单目摄像头采集到的手部关键点转换为机械臂末端目标位姿；
* 如何构建“手部输入—动作映射—控制器—MuJoCo 机械臂响应”的实时闭环。

当前 Retargeting 部分定位为**手部输入到 Franka Panda 末端位姿的动作映射原型**，不是完整的人体到人形机器人的全身动作重定向系统。

## 2. 功能模块

### 2.1 运动学与逆运动学

已实现内容：

* Franka Panda 七自由度机械臂模型加载与状态读取；
* 正运动学计算；
* 雅可比矩阵计算；
* 位置逆运动学；
* 六维位姿逆运动学；
* 旋转矩阵、欧拉角、四元数、旋转向量等姿态表示转换；
* 基于旋转向量的姿态误差计算；
* 末端目标位置和目标姿态跟踪验证。

核心思想：

* 通过正运动学计算末端位姿；
* 通过雅可比矩阵建立关节速度与末端速度之间的映射；
* 使用伪逆或阻尼伪逆求解关节增量；
* 在六维位姿控制中同时考虑位置误差和姿态误差。

### 2.2 冗余控制

Franka Panda 是七自由度机械臂，在完成六维末端任务时仍具有一定冗余自由度。本项目实现了基于零空间投影的冗余控制框架。

已实现内容：

* 奇异值分解；
* 奇异位形分析；
* 可操作度指标计算；
* 零空间投影；
* 任务优先级控制；
* 关节中心优化；
* 可操作度优化。

核心思想：

* 主任务优先保证末端位置或位姿收敛；
* 二级任务通过零空间投影进入控制，不破坏主任务；
* 通过关节中心优化降低靠近关节限位的风险；
* 通过可操作度优化改善机械臂在局部构型下的运动能力。

### 2.3 动力学控制

在运动学控制基础上，本项目进一步加入动力学控制模块，用于验证力矩级控制效果。

已实现内容：

* 关节空间 PD 控制；
* PD + 重力补偿；
* 基于 MuJoCo 逆动力学的力矩计算；
* 任务空间 PD / 阻抗控制；
* 雅可比转置力矩映射；
* 控制力矩限制；
* 目标位姿、实际位姿、跟踪误差和控制力矩记录。

当前任务空间控制逻辑：

```text
目标末端位置 / 姿态
        ↓
末端位置误差 / 姿态误差
        ↓
任务空间虚拟力 / 虚拟力矩
        ↓
雅可比转置映射
        ↓
关节力矩
        ↓
MuJoCo 仿真
```

### 2.4 手部动作映射

本项目基于 MediaPipe Hands 搭建了单目手部关键点采集模块，并将手部状态映射为 Franka Panda 末端目标位姿。

已实现内容：

* 摄像头图像读取；
* MediaPipe Hands 手部关键点检测；
* 手腕位置提取；
* 手掌方向估计；
* 拇指—食指距离计算；
* 手部平移到机械臂末端位置的映射；
* 手掌方向到机械臂末端姿态的映射；
* 捏合距离到夹爪开合的映射接口；
* 实时手部输入到 MuJoCo 机械臂响应的闭环验证。

当前 Retargeting 映射方式：

```text
手腕在图像中的左右移动 → 机械臂末端 y 方向移动
手腕在图像中的上下移动 → 机械臂末端 z 方向移动
手掌方向变化             → 机械臂末端姿态变化
拇指—食指距离             → 夹爪开合指令
```

当前为了保证演示稳定，平移控制主要验证 y / z 平面运动，深度方向和完整姿态映射仍在继续调试和优化中。

## 3. 项目结构

```text
robot-control-retargeting-framework/
├── core/
│   ├── kinematics.py
│   ├── jacobian.py
│   ├── pose_ik.py
│   ├── dynamics_control.py
│   └── filters.py
│
├── retargeting/
│   └── hand_to_panda.py
│
├── vision/
│   └── hand_tracker.py
│
├── demos/
│   ├── panda/
│   │   ├── demo_joint_pd_gc.py
│   │   ├── demo_task_space_impedance.py
│   │   └── demo_hand_retargeting_pd_gc.py
│   │
│   └── retargeting/
│       └── demo_hand_tracker_only.py
│
├── models/
│   └── panda/
│       └── panda.xml
│
├── scripts/
│   ├── plot_dynamics_results.py
│   └── plot_retargeting_results.py
│
├── results/
│   ├── dynamics/
│   └── retargeting/
│
├── README.md
└── requirements.txt
```

说明：

* `core/`：机器人运动学、逆运动学、动力学控制等核心算法；
* `vision/`：摄像头和 MediaPipe 手部关键点检测；
* `retargeting/`：手部动作到机械臂末端目标的映射；
* `demos/`：各类算法验证和实时演示脚本；
* `scripts/`：实验数据绘图与结果统计；
* `results/`：保存实验曲线、误差数据、控制力矩数据和演示结果。

## 4. 环境配置

### 4.1 Python 环境

建议使用 Python 3.10 或 3.11。

```bash
conda create -n humanoid python=3.10
conda activate humanoid
```

### 4.2 安装依赖

```bash
pip install mujoco numpy scipy matplotlib opencv-python mediapipe
```

或者使用：

```bash
pip install -r requirements.txt
```

推荐的 `requirements.txt`：

```text
mujoco
numpy
scipy
matplotlib
opencv-python
mediapipe
```

### 4.3 Windows PowerShell 运行说明

如果直接运行脚本时出现：

```text
ModuleNotFoundError: No module named 'core'
```

推荐在项目根目录使用 `-m` 方式运行，例如：

```powershell
D:\anaconda3\envs\humanoid\python.exe -m demos.panda.demo_task_space_impedance
```

也可以临时设置项目根目录到 `PYTHONPATH`：

```powershell
$env:PYTHONPATH="E:\运控\robot_control_retargeting_framework"
```

## 5. 快速运行

### 5.1 关节空间 PD 与重力补偿验证

运行：

```bash
python -m demos.panda.demo_joint_pd_gc --model models/panda/panda.xml
```

该实验用于对比：

* 关节空间 PD 控制；
* PD + 重力补偿控制。

输出数据：

```text
results/dynamics/joint_pd_only.csv
results/dynamics/joint_pd_gc.csv
```

### 5.2 任务空间 PD / 阻抗控制验证

运行：

```bash
python -m demos.panda.demo_task_space_impedance --model models/panda/panda.xml --body hand
```

该实验用于验证：

* 末端位置跟踪；
* 末端姿态保持；
* 雅可比转置力矩映射；
* 任务空间控制效果。

输出数据：

```text
results/dynamics/task_space_impedance.csv
```

### 5.3 绘制动力学控制结果

运行：

```bash
python scripts/plot_dynamics_results.py
```

输出图像和指标：

```text
results/dynamics/pd_gc_tracking_curve.png
results/dynamics/pd_gc_torque_curve.png
results/dynamics/task_space_impedance_error_curve.png
results/dynamics/metrics.csv
```

### 5.4 单独测试摄像头与手部关键点检测

运行：

```bash
python -m demos.retargeting.demo_hand_tracker_only
```

正常情况下会打开摄像头窗口，并显示手部关键点检测结果。

如果摄像头无法打开，可以尝试修改代码中的：

```python
camera_id=0
```

为：

```python
camera_id=1
```

或：

```python
camera_id=2
```

### 5.5 运行手部动作映射实时闭环

运行：

```bash
python -m demos.panda.demo_hand_retargeting_pd_gc
```

该实验会同时启动：

* MediaPipe 摄像头窗口；
* MuJoCo 机械臂仿真窗口。

实时闭环流程：

```text
摄像头采集手部图像
        ↓
MediaPipe 检测 21 个手部关键点
        ↓
提取手腕位置、手掌方向和捏合距离
        ↓
映射为 Franka Panda 末端目标位姿
        ↓
任务空间 PD / 阻抗控制器生成关节力矩
        ↓
MuJoCo 中机械臂响应
```

## 6. 实验结果

### 6.1 关节空间 PD 与重力补偿

通过关节目标阶跃实验，对比 PD 与 PD + 重力补偿的收敛效果。

重点观察指标：

* 关节误差范数；
* 稳态误差；
* 控制力矩范数；
* 是否出现明显振荡。

预期现象：

* 加入重力补偿后，关节稳态误差降低；
* 抗重力方向的关节控制效果更稳定；
* 控制器能够在合理力矩范围内完成目标关节角跟踪。

### 6.2 任务空间控制

通过给定末端目标轨迹，验证任务空间 PD / 阻抗控制器对末端位置和姿态的跟踪能力。

重点观察指标：

* 末端位置误差；
* 末端姿态误差；
* 控制力矩；
* 机械臂运动是否平稳。

当前控制方式：

```text
末端误差 → 任务空间虚拟力 → 雅可比转置 → 关节力矩
```

### 6.3 手部动作映射

基于 MediaPipe Hands 的实时手部输入，验证手部动作到机械臂末端目标位姿的映射关系。

当前已验证内容：

* 手部左右移动能够驱动机械臂末端 y 方向运动；
* 手部上下移动能够驱动机械臂末端 z 方向运动；
* 通过低通滤波减少关键点抖动；
* 通过限制目标工作空间提高实时控制稳定性；
* 通过目标位姿、实际位姿、误差和力矩日志进行参数整定。

当前调试中重点关注：

* 末端 z 方向跟踪稳定性；
* 姿态控制与平移控制之间的耦合；
* 深度方向映射；
* 夹爪开合接口与视觉输入之间的联动。

## 7. 当前局限

当前项目仍属于算法验证与仿真原型阶段，存在以下局限：

1. **Retargeting 范围有限**
   当前实现的是手部输入到 Franka Panda 末端目标位姿的动作映射，不是完整的人体骨架到人形机器人的全身动作重定向。

2. **深度方向映射仍需优化**
   单目摄像头下深度估计不稳定，当前主要验证图像平面内的左右和上下移动。

3. **姿态映射仍在调试**
   手掌方向到机械臂末端姿态的映射关系受坐标系定义、手部检测稳定性和末端姿态约束影响，目前仍需要进一步标定。

4. **控制器仍以仿真验证为主**
   当前控制器主要在 MuJoCo 中验证，尚未部署到真实机器人。

5. **暂未实现完整全身控制**
   当前未实现人形机器人的全身运动控制、接触约束、质心控制、全身 QP 控制等模块。

## 8. 后续计划

### 8.1 动力学控制方向

* 完善关节空间 PD、PD + 重力补偿和逆动力学控制对比实验；
* 增加任务空间速度和加速度前馈；
* 加入更完整的阻抗控制和导纳控制实验；
* 记录并分析末端误差、关节力矩、控制频率和稳定性指标；
* 尝试加入关节力矩限制、速度限制和安全工作空间约束。

### 8.2 手部动作映射方向

* 完善手部平移、姿态和捏合动作到机械臂末端位姿及夹爪开合的映射；
* 增加坐标系标定流程；
* 增加关键点滤波、异常点剔除和轨迹平滑；
* 增加 Retargeting 误差曲线和实时延迟统计；
* 将夹爪开合动作接入 MuJoCo 夹爪模型。

### 8.3 工程化方向

* 重构核心算法接口；
* 增加配置文件管理控制参数；
* 增加实验日志自动保存；
* 增加单元测试；
* 整理演示视频和实验曲线；
* 完善 README 和代码注释。

### 8.4 长期方向

* 接入人体骨架数据或 VR 轨迹；
* 扩展到双臂或人形机器人模型；
* 尝试基于优化的动作重定向方法；
* 探索全身控制、接触约束和 QP 控制；
* 为真实机器人部署预留接口。

## 9. 项目关键词

* 机器人运动控制
* MuJoCo 仿真
* Franka Panda 七自由度机械臂
* 正运动学
* 雅可比矩阵
* 逆运动学
* 位姿控制
* 零空间控制
* 任务优先级控制
* 可操作度优化
* PD 控制
* 重力补偿
* 任务空间控制
* 阻抗控制
* MediaPipe Hands
* 手部动作映射
* Motion Retargeting
* 遥操作控制

## 10. 项目状态

当前项目仍在持续开发中。现阶段重点是完善动力学控制实验、优化手部动作映射稳定性，并整理可复现实验结果。

已完成：

* Franka Panda MuJoCo 仿真环境；
* 运动学与逆运动学算法；
* 冗余控制算法；
* 关节空间 PD 与重力补偿；
* 任务空间控制器；
* MediaPipe 手部关键点采集；
* 手部输入到机械臂末端位姿的实时闭环原型。

进行中：

* 姿态映射稳定性优化；
* z 方向跟踪效果优化；
* Retargeting 实验数据记录；
* README、实验结果和演示材料整理。

## 11. 参考说明

本项目主要用于个人学习、机器人运动控制算法验证和求职项目展示。项目重点不在于完成工业级机器人控制系统，而在于通过可运行的代码和可复现实验展示以下能力：

* 机器人运动学与控制基础；
* MuJoCo 仿真与调试能力；
* 任务空间控制和力矩控制理解；
* 手部视觉输入处理与动作映射设计；
* 从算法 Demo 到可复用工程框架的整理能力。

