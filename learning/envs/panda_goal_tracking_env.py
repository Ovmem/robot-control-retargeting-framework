from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - exercised only without optional deps
    raise ImportError(
        "PandaGoalTrackingEnv requires gymnasium. Install RL dependencies with: "
        "pip install gymnasium stable-baselines3 pyyaml tensorboard"
    ) from exc

from core.dynamics_control import (
    PandaTorqueController,
    TorqueLimit,
    get_body_pose,
    has_affine_position_actuators,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class PandaGoalTrackingConfig:
    model_path: str = "models/panda/panda_torque.xml"
    body_name: str = "hand"
    max_episode_steps: int = 160
    control_decimation: int = 10
    action_scale: float = 0.035
    joint_target_clip: float = 0.08
    success_threshold: float = 0.035
    success_hold_steps: int = 6
    reset_noise: float = 0.05
    workspace_radius: float = 0.16
    workspace_z_min: float = 0.22
    workspace_z_max: float = 0.78
    q_home: list[float] = field(
        default_factory=lambda: [0.0, -0.7, 0.0, -2.2, 0.0, 1.6, 0.8]
    )
    kp: float | list[float] = field(
        default_factory=lambda: [140.0, 140.0, 120.0, 110.0, 45.0, 35.0, 25.0]
    )
    kd: float | list[float] = field(
        default_factory=lambda: [18.0, 18.0, 16.0, 14.0, 6.0, 5.0, 4.0]
    )
    torque_limit: float | list[float] = field(
        default_factory=lambda: [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]
    )
    q_limit_margin: float = 0.05
    terminate_on_joint_limit: bool = True
    debug: bool = False


@dataclass
class RewardConfig:
    w_position: float = 2.0
    k_position: float = 35.0
    w_success: float = 8.0
    w_action: float = 0.01
    w_action_delta: float = 0.005
    w_torque: float = 0.00002
    w_joint_limit: float = 0.05


class PandaGoalTrackingEnv(gym.Env[np.ndarray, np.ndarray]):
    """Task-conditioned Panda end-effector position reaching environment."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        config: dict[str, Any] | PandaGoalTrackingConfig | None = None,
        reward_config: dict[str, Any] | RewardConfig | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()

        self.cfg = self._make_config(config, PandaGoalTrackingConfig)
        self.reward_cfg = self._make_config(reward_config, RewardConfig)
        self.render_mode = render_mode
        self.dof = 7

        model_path = Path(self.cfg.model_path)
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path
        self.model_path = model_path
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        if has_affine_position_actuators(self.model, dof=self.dof):
            raise RuntimeError(
                "PandaGoalTrackingEnv requires torque actuators. Use models/panda/panda_torque.xml."
            )

        self.q_home = np.asarray(self.cfg.q_home, dtype=np.float64)
        self.kp = self._as_dof_array(self.cfg.kp)
        self.kd = self._as_dof_array(self.cfg.kd)
        tau_limit = self._as_dof_array(self.cfg.torque_limit)
        self.torque_limit = TorqueLimit(lower=-tau_limit, upper=tau_limit)
        self.controller = PandaTorqueController(
            self.model,
            self.data,
            dof=self.dof,
            body_name=self.cfg.body_name,
            torque_limit=self.torque_limit,
        )

        self.joint_lower, self.joint_upper = self._joint_ranges()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.dof,), dtype=np.float32)
        obs_limit = np.full(30, 10.0, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_limit, obs_limit, dtype=np.float32)

        self.viewer = None
        self.renderer = None
        self.goal_position = np.zeros(3, dtype=np.float64)
        self.q_target = self.q_home.copy()
        self.previous_action = np.zeros(self.dof, dtype=np.float64)
        self.last_action_delta_norm = 0.0
        self.last_action_clipped_fraction = 0.0
        self.last_tau_clipped_fraction = 0.0
        self.last_tau = np.zeros(self.dof, dtype=np.float64)
        self.step_count = 0
        self.success_hold_count = 0

    @staticmethod
    def _make_config(value: Any, cls: type) -> Any:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        return cls(**value)

    def _as_dof_array(self, value: float | list[float]) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 0:
            arr = np.full(self.dof, float(arr), dtype=np.float64)
        if arr.shape != (self.dof,):
            raise ValueError(f"Expected scalar or {self.dof}-vector, got shape {arr.shape}.")
        return arr

    def _joint_ranges(self) -> tuple[np.ndarray, np.ndarray]:
        lower = np.empty(self.dof, dtype=np.float64)
        upper = np.empty(self.dof, dtype=np.float64)
        for i in range(self.dof):
            joint_id = self.model.joint(f"joint{i + 1}").id
            lower[i], upper[i] = self.model.jnt_range[joint_id]
        return lower, upper

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}

        q_noise = self.np_random.uniform(-self.cfg.reset_noise, self.cfg.reset_noise, size=self.dof)
        q0 = np.clip(self.q_home + q_noise, self.joint_lower + 0.02, self.joint_upper - 0.02)

        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        self.data.qpos[: self.dof] = q0
        if self.model.nq >= 9:
            self.data.qpos[7:9] = 0.04
        mujoco.mj_forward(self.model, self.data)

        self.q_target = q0.copy()
        self.previous_action = np.zeros(self.dof, dtype=np.float64)
        self.last_action_delta_norm = 0.0
        self.last_action_clipped_fraction = 0.0
        self.last_tau_clipped_fraction = 0.0
        self.last_tau = np.zeros(self.dof, dtype=np.float64)
        self.step_count = 0
        self.success_hold_count = 0

        initial_ee_position = self.ee_position()
        self.goal_position = np.asarray(
            options.get("goal_position", self._sample_goal_position()), dtype=np.float64
        )

        obs = self._get_obs()
        info = self._get_info(reward_terms=self._zero_reward_terms())
        info.update(
            {
                "initial_ee_position": initial_ee_position.astype(np.float32),
                "initial_goal_distance": float(np.linalg.norm(self.goal_position - initial_ee_position)),
                "joint_target": self.q_target.astype(np.float32),
            }
        )
        if self.cfg.debug:
            self._print_reset_debug(info)
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action_raw = np.asarray(action, dtype=np.float64).reshape(self.dof)
        clipped_action = np.clip(action_raw, -1.0, 1.0)
        prev_action = self.previous_action.copy()
        self.last_action_clipped_fraction = float(np.mean(np.abs(action_raw - clipped_action) > 1e-8))

        delta_q = np.clip(
            clipped_action * self.cfg.action_scale,
            -self.cfg.joint_target_clip,
            self.cfg.joint_target_clip,
        )
        self.q_target = np.clip(
            self.q_target + delta_q,
            self.joint_lower + self.cfg.q_limit_margin,
            self.joint_upper - self.cfg.q_limit_margin,
        )

        for _ in range(int(self.cfg.control_decimation)):
            tau_raw = self._joint_pd_gravity_comp_unclipped(self.q_target)
            tau = self.controller.clip_tau(tau_raw)
            self.last_tau_clipped_fraction = float(np.mean(np.abs(tau_raw - tau) > 1e-8))
            # Keep the executable control path aligned with PandaTorqueController.
            tau = self.controller.joint_pd_gravity_comp(
                q_des=self.q_target,
                kp=self.kp,
                kd=self.kd,
                gravity_comp=True,
            )
            self.last_tau = self.controller.apply_torque(tau, prefer_ctrl=True)
            if self.model.nu >= 8:
                self.data.ctrl[7] = 0.0
            mujoco.mj_step(self.model, self.data)

        self.previous_action = clipped_action.copy()
        self.last_action_delta_norm = float(np.linalg.norm(clipped_action - prev_action))
        self.step_count += 1
        reward, terms = self._compute_reward(clipped_action, prev_action)

        error = float(np.linalg.norm(self.goal_position - self.ee_position()))
        if error < self.cfg.success_threshold:
            self.success_hold_count += 1
        else:
            self.success_hold_count = 0

        success = self.success_hold_count >= int(self.cfg.success_hold_steps)
        failure_reason = self._failure_reason()
        failed = failure_reason is not None
        terminated = bool(success or failed)
        truncated = bool(self.step_count >= int(self.cfg.max_episode_steps) and not terminated)

        obs = self._get_obs()
        info = self._get_info(reward_terms=terms)
        info.update(
            {
                "is_success": bool(success),
                "failed": bool(failed),
                "failure_reason": failure_reason,
                "action_clipped": bool(np.any(np.abs(action_raw - clipped_action) > 1e-8)),
                "applied_action": clipped_action.astype(np.float32),
                "q_target": self.q_target.astype(np.float32),
            }
        )
        return obs, float(reward), terminated, truncated, info

    def _sample_goal_position(self) -> np.ndarray:
        home_pos = self.ee_position()
        for _ in range(100):
            offset = self.np_random.uniform(-self.cfg.workspace_radius, self.cfg.workspace_radius, size=3)
            if np.linalg.norm(offset) > self.cfg.workspace_radius:
                continue
            candidate = home_pos + offset
            candidate[2] = np.clip(candidate[2], self.cfg.workspace_z_min, self.cfg.workspace_z_max)
            if np.linalg.norm(candidate - home_pos) <= self.cfg.workspace_radius:
                return candidate.astype(np.float64)
        return home_pos + np.array([0.08, 0.0, 0.02], dtype=np.float64)

    def ee_position(self) -> np.ndarray:
        pos, _ = get_body_pose(self.model, self.data, self.cfg.body_name)
        return pos

    def joint_names(self) -> list[str]:
        return [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.model.joint(f"joint{i + 1}").id) for i in range(self.dof)]

    def actuator_names(self) -> list[str]:
        names = []
        for i in range(min(self.dof, self.model.nu)):
            names.append(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"actuator_{i}")
        return names

    def _joint_pd_gravity_comp_unclipped(self, q_des: np.ndarray) -> np.ndarray:
        tau = self.kp * (np.asarray(q_des, dtype=np.float64)[: self.dof] - self.controller.q())
        tau = tau - self.kd * self.controller.qd()
        return tau + self.controller.gravity_comp_torque()

    def _get_obs(self) -> np.ndarray:
        q = np.clip(self.data.qpos[: self.dof].copy(), self.joint_lower, self.joint_upper)
        dq = np.clip(self.data.qvel[: self.dof].copy(), -10.0, 10.0)
        ee = np.clip(self.ee_position(), -2.0, 2.0)
        goal = np.clip(self.goal_position, -2.0, 2.0)
        rel = np.clip(goal - ee, -1.0, 1.0)
        obs = np.concatenate([q, dq, ee, goal, rel, self.previous_action])
        return np.clip(obs, -10.0, 10.0).astype(np.float32)

    def _compute_reward(
        self,
        action: np.ndarray,
        previous_action: np.ndarray,
    ) -> tuple[float, dict[str, float]]:
        err = float(np.linalg.norm(self.goal_position - self.ee_position()))
        position_reward = self.reward_cfg.w_position * float(
            np.exp(-self.reward_cfg.k_position * err * err)
        )
        success_bonus = self.reward_cfg.w_success if err < self.cfg.success_threshold else 0.0
        action_penalty = self.reward_cfg.w_action * float(np.sum(np.square(action)))
        action_delta_penalty = self.reward_cfg.w_action_delta * float(
            np.sum(np.square(action - previous_action))
        )
        torque_penalty = self.reward_cfg.w_torque * float(np.sum(np.square(self.last_tau)))
        joint_limit_penalty_raw = self._joint_limit_penalty()
        joint_limit_penalty = self.reward_cfg.w_joint_limit * joint_limit_penalty_raw
        total = (
            position_reward
            + success_bonus
            - action_penalty
            - action_delta_penalty
            - torque_penalty
            - joint_limit_penalty
        )
        terms = {
            "reward_position": position_reward,
            "reward_success": success_bonus,
            "penalty_action": action_penalty,
            "penalty_action_delta": action_delta_penalty,
            "penalty_torque": torque_penalty,
            "penalty_joint_limit": joint_limit_penalty,
            "joint_limit_penalty_raw": joint_limit_penalty_raw,
        }
        return float(total), terms

    def _joint_limit_penalty(self) -> float:
        q = self.data.qpos[: self.dof]
        dist_to_limits = np.minimum(q - self.joint_lower, self.joint_upper - q)
        margin = max(float(self.cfg.q_limit_margin), 1e-6)
        return float(np.sum(np.square(np.clip((margin - dist_to_limits) / margin, 0.0, 1.0))))

    def _failure_reason(self) -> str | None:
        q = self.data.qpos[: self.dof]
        dq = self.data.qvel[: self.dof]
        ee = self.ee_position()
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(dq)) or not np.all(np.isfinite(ee)):
            return "nan_state"
        if not np.all(np.isfinite(self.last_tau)):
            return "numerical_instability"
        if np.max(np.abs(dq)) > 40.0:
            return "excessive_joint_velocity"
        if self.cfg.terminate_on_joint_limit:
            if np.any(q < self.joint_lower - 0.05) or np.any(q > self.joint_upper + 0.05):
                return "severe_joint_limit_violation"
        if np.max(np.abs(q)) > 1e4 or np.max(np.abs(dq)) > 1e4 or np.max(np.abs(self.last_tau)) > 1e5:
            return "numerical_instability"
        return None

    def _joint_limit_margin_min(self) -> float:
        q = self.data.qpos[: self.dof]
        return float(np.min(np.minimum(q - self.joint_lower, self.joint_upper - q)))

    def _print_reset_debug(self, info: dict[str, Any]) -> None:
        q = self.data.qpos[: self.dof]
        dq = self.data.qvel[: self.dof]
        print("[PandaGoalTrackingEnv debug reset]")
        print(f"  model_path={self.model_path}")
        print(f"  initial_ee_position={info['initial_ee_position']}")
        print(f"  goal_position={info['goal_position']}")
        print(f"  initial_goal_distance={info['initial_goal_distance']:.6f}")
        print(f"  q_minmax=({float(np.min(q)):.4f}, {float(np.max(q)):.4f})")
        print(f"  dq_norm={float(np.linalg.norm(dq)):.6f}")
        print(f"  joint_target={info['joint_target']}")

    def _zero_reward_terms(self) -> dict[str, float]:
        return {
            "reward_position": 0.0,
            "reward_success": 0.0,
            "penalty_action": 0.0,
            "penalty_action_delta": 0.0,
            "penalty_torque": 0.0,
            "penalty_joint_limit": 0.0,
            "joint_limit_penalty_raw": 0.0,
        }

    def _get_info(self, reward_terms: dict[str, float]) -> dict[str, Any]:
        ee = self.ee_position()
        err = float(np.linalg.norm(self.goal_position - ee))
        info: dict[str, Any] = {
            "ee_position": ee.astype(np.float32),
            "goal_position": self.goal_position.astype(np.float32),
            "position_error": err,
            "success_hold_count": int(self.success_hold_count),
            "step_count": int(self.step_count),
            "tau_norm": float(np.linalg.norm(self.last_tau)),
            "torque_norm": float(np.linalg.norm(self.last_tau)),
            "tau_clipped_fraction": float(self.last_tau_clipped_fraction),
            "max_abs_torque": float(np.max(np.abs(self.last_tau))),
            "max_abs_joint_velocity": float(np.max(np.abs(self.data.qvel[: self.dof]))),
            "joint_limit_margin_min": self._joint_limit_margin_min(),
            "action_clipped_fraction": float(self.last_action_clipped_fraction),
            "failure_reason": None,
            "mean_action_delta": float(self.last_action_delta_norm),
        }
        info.update(reward_terms)
        return info

    def render(self):
        if self.render_mode == "human":
            import mujoco.viewer as mujoco_viewer

            if self.viewer is None:
                self.viewer = mujoco_viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
            return None
        if self.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            self.renderer.update_scene(self.data)
            return self.renderer.render()
        return None

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
