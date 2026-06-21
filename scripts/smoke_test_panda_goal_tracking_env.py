from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from learning.envs import PandaGoalTrackingEnv  # noqa: E402
from learning.utils.run_utils import load_yaml_config  # noqa: E402


REWARD_KEYS = [
    "reward_position",
    "reward_success",
    "penalty_action",
    "penalty_action_delta",
    "penalty_torque",
    "penalty_joint_limit",
]


def assert_finite(name: str, value) -> None:
    arr = np.asarray(value)
    if not np.all(np.isfinite(arr)):
        raise RuntimeError(f"Non-finite value detected in {name}: {value}")


def run_sb3_checker(env: PandaGoalTrackingEnv) -> None:
    try:
        from stable_baselines3.common.env_checker import check_env

        check_env(env, warn=True, skip_render_check=True)
        print("SB3 check_env: passed")
    except Exception as exc:
        print("SB3 check_env: failed")
        print(f"  model_path={env.model_path}")
        print(f"  observation_space={env.observation_space}")
        print(f"  action_space={env.action_space}")
        print(f"  joints={env.joint_names()}")
        print(f"  actuators={env.actuator_names()}")
        raise RuntimeError("Stable-Baselines3 environment compatibility check failed.") from exc


def main() -> None:
    config = load_yaml_config("learning/configs/panda_ppo_default.yaml")
    env = PandaGoalTrackingEnv(config=config.get("env", {}), reward_config=config.get("reward", {}))

    print("=== Panda Goal Tracking Smoke Test ===")
    print(f"model_path={env.model_path}")
    print(f"controlled_dof={env.dof}")
    print(f"joint_names={env.joint_names()}")
    print(f"actuator_names={env.actuator_names()}")
    print(f"observation_space={env.observation_space}")
    print(f"action_space={env.action_space}")
    print(f"control_decimation={env.cfg.control_decimation}")

    run_sb3_checker(env)

    obs, info = env.reset(seed=0)
    assert_finite("reset observation", obs)
    assert_finite("reset info ee_position", info["ee_position"])
    assert_finite("reset info goal_position", info["goal_position"])

    q = obs[:7]
    dq = obs[7:14]
    ee_position = obs[14:17]
    goal_position = obs[17:20]

    print(f"reset obs_shape={obs.shape} contains={env.observation_space.contains(obs)}")
    print(
        "reset shapes "
        f"q={q.shape} dq={dq.shape} ee_position={ee_position.shape} "
        f"goal_position={goal_position.shape} tau={env.last_tau.shape}"
    )
    print(
        "reset info "
        f"initial_goal_distance={info['initial_goal_distance']:.4f} "
        f"position_error={info['position_error']:.4f} "
        f"joint_target_shape={info['joint_target'].shape}"
    )

    total_reward = 0.0
    for step in range(20):
        action = env.action_space.sample()
        assert_finite("sampled action", action)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        assert_finite("step observation", obs)
        assert_finite("step reward", reward)
        assert_finite("step tau_norm", info["tau_norm"])
        assert_finite("step position_error", info["position_error"])
        for key in REWARD_KEYS:
            assert_finite(key, info[key])

        action_norm = float(np.linalg.norm(action))
        print(
            f"step={step:02d} reward={reward:.4f} total={total_reward:.4f} "
            f"terminated={terminated} truncated={truncated} "
            f"error={info['position_error']:.4f} tau={info['tau_norm']:.3f} "
            f"|a|={action_norm:.3f} d_action={info['mean_action_delta']:.3f} "
            f"tau_clip={info['tau_clipped_fraction']:.2f} "
            f"action_clip={info['action_clipped_fraction']:.2f} "
            f"joint_margin={info['joint_limit_margin_min']:.4f} "
            f"failure={info['failure_reason']} "
            f"terms={{pos:{info['reward_position']:.3f}, success:{info['reward_success']:.3f}, "
            f"act:{info['penalty_action']:.3f}, dact:{info['penalty_action_delta']:.3f}, "
            f"tau:{info['penalty_torque']:.3f}, joint:{info['penalty_joint_limit']:.3f}}}"
        )
        if terminated or truncated:
            break

    env.close()
    print("Smoke test complete.")


if __name__ == "__main__":
    main()
