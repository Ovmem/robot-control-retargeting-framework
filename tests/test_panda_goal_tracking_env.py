from __future__ import annotations

import numpy as np
import pytest

gymnasium = pytest.importorskip("gymnasium")

from gymnasium.utils.env_checker import check_env  # noqa: E402

from learning.envs import PandaGoalTrackingEnv  # noqa: E402
from learning.utils.run_utils import load_yaml_config  # noqa: E402


def make_env() -> PandaGoalTrackingEnv:
    config = load_yaml_config("learning/configs/panda_ppo_default.yaml")
    env_config = config.get("env", {})
    env_config["max_episode_steps"] = 20
    return PandaGoalTrackingEnv(config=env_config, reward_config=config.get("reward", {}))


@pytest.mark.mujoco
def test_reset_observation_matches_space() -> None:
    env = make_env()
    obs, info = env.reset(seed=0)
    assert obs.shape == (30,)
    assert env.observation_space.contains(obs)
    assert "position_error" in info
    env.close()


@pytest.mark.mujoco
def test_step_types_and_no_nan() -> None:
    env = make_env()
    env.reset(seed=1)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert np.all(np.isfinite(obs))
    assert np.isfinite(reward)
    assert np.isfinite(info["position_error"])
    env.close()


@pytest.mark.mujoco
def test_action_clipping_and_reward_terms() -> None:
    env = make_env()
    env.reset(seed=2)
    action = np.full(7, 5.0, dtype=np.float32)
    _, _, _, _, info = env.step(action)
    assert info["action_clipped"] is True
    assert np.all(np.abs(info["applied_action"]) <= 1.0)
    for key in [
        "reward_position",
        "reward_success",
        "penalty_action",
        "penalty_action_delta",
        "penalty_torque",
        "penalty_joint_limit",
        "tau_norm",
        "tau_clipped_fraction",
        "max_abs_joint_velocity",
        "joint_limit_margin_min",
        "action_clipped_fraction",
        "failure_reason",
    ]:
        assert key in info
    env.close()


@pytest.mark.mujoco
def test_gymnasium_checker() -> None:
    env = make_env()
    check_env(env, skip_render_check=True)
    env.close()
