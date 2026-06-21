# Task-Conditioned Panda Goal Tracking with PPO

This extension adds a first-stage reinforcement learning loop for task-conditioned Franka Panda end-effector position reaching in MuJoCo. The policy observes the robot state and a sampled goal position, then outputs small joint target increments. A joint-space PD controller with MuJoCo gravity compensation converts those targets to torque commands for the torque-actuated Panda model.

Current scope boundaries:

- It tracks end-effector position only; the end-effector pose orientation is fixed by the underlying Panda state and is not trained as a goal.
- It trains in simulation only.
- It is not a direct torque policy; PPO outputs joint target increments.
- It is not a humanoid whole-body policy.
- Hand-action CSV trajectory replay and hand retargeting integration are next-stage work.

## Control Structure

```text
goal position
-> PPO policy
-> joint target increment
-> PD + gravity compensation
-> MuJoCo Panda
-> state / reward
```

The environment uses `models/panda/panda_torque.xml`, the first 7 Panda joints, and the existing `core.dynamics_control.PandaTorqueController`. Each PPO action is repeated for `control_decimation` MuJoCo physics steps.

## Observation

The observation is a 30-dimensional `float32` vector:

```text
q                       7 current joint angles
dq                      7 current joint velocities, clipped for stability
ee_position             3 current hand body position in world coordinates
goal_position           3 sampled reachable goal position
goal_relative_position  3 goal_position - ee_position
previous_action         7 previous clipped policy action
```

The observation space is a finite Gymnasium `Box` and values are clipped to avoid numeric explosions. It contains only current or past information.

## Action

The action space is a 7-dimensional continuous Gymnasium `Box(-1, 1)`. The environment interprets it as normalized joint target increments:

```text
delta_q_target = clip(action, -1, 1) * action_scale
q_target = clip(q_target + delta_q_target, joint limits with margin)
```

Key configurable control parameters live in `learning/configs/panda_ppo_default.yaml`:

- `action_scale`
- `joint_target_clip`
- `control_decimation`
- `kp`
- `kd`
- `torque_limit`

All policy actions, accumulated joint targets, and torque commands are clipped.

## Reward

The reward is decomposed and returned in every `info` dict:

```text
r_total =
  w_position * exp(-k_position * ||ee_position - goal_position||^2)
+ w_success * success_bonus
- w_action * ||action||^2
- w_action_delta * ||action - previous_action||^2
- w_torque * ||tau||^2
- w_joint_limit * joint_limit_penalty
```

- `reward_position` encourages the end-effector to approach the sampled goal.
- `reward_success` adds a bonus inside `success_threshold`.
- `penalty_action` discourages large target jumps.
- `penalty_action_delta` discourages action jitter.
- `penalty_torque` discourages excessive torque use.
- `penalty_joint_limit` discourages sitting near joint limits.

The YAML weights are initial configuration values, not optimized benchmark settings.

## Termination

An episode ends when one of the following occurs:

- Success: position error stays below `success_threshold` for `success_hold_steps` control steps.
- Truncation: `max_episode_steps` is reached.
- Failure: NaN, severe velocity/state abnormality, or serious joint-limit violation occurs.

Large position error alone is not treated as failure, so PPO can explore.

## Installation

Install the RL dependencies in addition to the base MuJoCo stack:

```bash
pip install gymnasium stable-baselines3 pyyaml tensorboard
```

The repository `requirements.txt` also lists `torch`; Stable-Baselines3 may install a compatible PyTorch build automatically, but on CUDA systems you may prefer the official PyTorch install command for your hardware.

## Smoke Test

```bash
python scripts/smoke_test_panda_goal_tracking_env.py
```

The smoke test prints the resolved model path, controlled joint and actuator names, action and observation spaces, compact shape checks, Stable-Baselines3 `check_env` status, per-step reward terms, position error, torque norm, action norm, action delta norm, clipping fractions, and non-finite value checks. It does not require a viewer, camera, or GUI.

## First Run Checklist

1. Install dependencies.
2. Run smoke test.
3. Run pytest.
4. Run 1000-step PPO smoke training.
5. Evaluate the saved model for 5 episodes.
6. Only after all checks pass, start longer training.

## Training

PowerShell single-line command:

```powershell
python scripts/train_panda_goal_tracking_ppo.py --total-timesteps 1000 --seed 0 --run-name smoke_test
```

PowerShell multiline command:

```powershell
python scripts/train_panda_goal_tracking_ppo.py `
  --total-timesteps 1000 `
  --seed 0 `
  --run-name smoke_test
```

Supported arguments:

- `--config`
- `--total-timesteps`
- `--seed`
- `--run-name`
- `--device`
- `--n-envs`
- `--resume`

Default training uses one environment for Windows compatibility. `--n-envs > 1` enables vectorized training.

Before PPO starts, the training script prints a configuration summary and runs a reset plus 3 random-action preflight. If the preflight sees non-finite observations, non-finite rewards, or an environment failure reason, training stops before any model files are written.

## Evaluation

PowerShell single-line command:

```powershell
python scripts/evaluate_panda_goal_tracking_ppo.py --model results/rl_tracking/runs/<run>/checkpoints/final_model.zip --episodes 5 --seed 0
```

PowerShell multiline command:

```powershell
python scripts/evaluate_panda_goal_tracking_ppo.py `
  --model results/rl_tracking/runs/<run>/checkpoints/final_model.zip `
  --episodes 5 `
  --seed 0
```

Evaluation writes `evaluation_episodes.csv` and `evaluation_summary.json`. Reported metrics include:

- `success_rate`
- `mean_final_position_error`
- `median_final_position_error`
- `mean_episode_return`
- `mean_episode_length`
- `rms_torque`
- `mean_action_delta`

Use `--render` only when a viewer is desired. Use `--record-video` only when video capture is intentionally enabled.

## Output Directory

Training writes to:

```text
results/rl_tracking/runs/<timestamp>_<run_name>/
|-- checkpoints/
|-- logs/
|-- metrics/
|-- config/
+-- videos/
```

Model zip files, TensorBoard event logs, and videos are ignored by git.

## Current Limitations

- No curriculum learning yet.
- No reward ablation study yet.
- No hand-action CSV trajectory replay yet.
- No joint position-plus-orientation goal tracking yet.
- No sim-to-real randomization yet.
- No claimed convergence curve, success-rate benchmark, or hardware result is included unless generated by an actual run.
