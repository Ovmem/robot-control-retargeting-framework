# 测试策略

项目使用 pytest 进行测试发现和执行。

## 自动化测试

自动化测试是满足以下条件的 pytest 测试：

- 不需要显示器、viewer GUI 或摄像头
- 可以无头运行（无 MuJoCo viewer）
- 在数秒内完成
- 包含 pose IK 收敛测试（tests/panda/test_pose_ik.py）
- 有明确的 assert 语句
- 包含 retargeting 映射测试（tests/panda/test_retargeting_mapping.py）

运行自动化测试（跳过 viewer/interactive 测试）：

\`\`\`bash
pytest -q -m "not viewer and not interactive"
\`\`\`

也可排除 MuJoCo 模型加载测试：

\`\`\`bash
pytest -q -m "not viewer and not interactive and not mujoco"
\`\`\`

仅运行 MuJoCo 相关测试：

\`\`\`bash
pytest -q -m mujoco
\`\`\`

坐标帧假设和映射约束详见 docs/retargeting.md。

## 交互式 / Viewer 验证

标记为 @pytest.mark.viewer 的测试需要：

- 显示器和 MuJoCo GUI viewer
- 人工视觉检查
- 通常涉及 mujoco.viewer.launch_passive()

这些测试在 pytest 或 pytest -m "not viewer and not interactive" 中默认跳过。

它们仍然可以手动作为独立脚本运行：

\`\`\`bash
python tests/panda/test_view_panda.py
python tests/panda/test_gravity_compensation.py
python tests/panda/test_pd_control.py
python tests/panda/test_pd_gravity_compensation.py
\`\`\`

## 手部 Retargeting Demo

demos/panda/demo_hand_retargeting_pd_gc.py 不是 pytest 测试。它需要：

- 可用的摄像头
- OpenCV GUI 窗口
- MuJoCo viewer 显示

坐标帧假设和映射约束详见 docs/retargeting.md。

## 自定义 Markers

注册在 pytest.ini 中：

| Marker | 说明 |
|---|---|
| interactive | 需要 viewer、GUI、摄像头或人工检查 |
| viewer | 需要 MuJoCo viewer 和显示器 |
| mujoco | 需要加载　MuJoCo 模型 |
| slow | 运行时间较长的验证 |
