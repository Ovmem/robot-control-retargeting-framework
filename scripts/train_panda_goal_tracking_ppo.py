from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from learning.envs import PandaGoalTrackingEnv  # noqa: E402
from learning.utils.run_utils import (  # noqa: E402
    copy_config,
    load_yaml_config,
    make_run_dir,
    save_json,
    set_global_seed,
)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PPO for Panda goal tracking.")
    parser.add_argument("--config", default="learning/configs/panda_ppo_default.yaml")
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", default="panda_goal_tracking")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--resume", default=None, help="Optional SB3 PPO zip path to resume.")
    return parser


def _make_env(env_config: dict, reward_config: dict, seed: int, rank: int = 0):
    def thunk():
        env = PandaGoalTrackingEnv(config=env_config, reward_config=reward_config)
        env.reset(seed=seed + rank)
        return env

    return thunk


def print_config_summary(env_config: dict, reward_config: dict) -> None:
    model_path = env_config.get("model_path", "models/panda/panda_torque.xml")
    control_decimation = int(env_config.get("control_decimation", 10))
    physics_dt = 0.0
    try:
        preflight_env = PandaGoalTrackingEnv(config=env_config, reward_config=reward_config)
        physics_dt = float(preflight_env.model.opt.timestep)
        preflight_env.close()
    except Exception:
        physics_dt = 0.0
    control_dt = physics_dt * control_decimation if physics_dt > 0.0 else None
    workspace = {
        "radius": env_config.get("workspace_radius"),
        "z_min": env_config.get("workspace_z_min"),
        "z_max": env_config.get("workspace_z_max"),
    }

    print("=== PPO training configuration ===")
    print(f"model_path={model_path}")
    print(f"control_dt={control_dt}")
    print(f"control_decimation={control_decimation}")
    print(f"max_episode_steps={env_config.get('max_episode_steps')}")
    print(f"action_scale={env_config.get('action_scale')}")
    print(f"joint_target_clip={env_config.get('joint_target_clip')}")
    print(f"kp={env_config.get('kp')}")
    print(f"kd={env_config.get('kd')}")
    print(f"torque_limit={env_config.get('torque_limit')}")
    print(f"goal_workspace={workspace}")
    print(f"success_threshold={env_config.get('success_threshold')}")
    print(f"reward_weights={reward_config}")


def run_preflight(env_config: dict, reward_config: dict, seed: int) -> None:
    env = PandaGoalTrackingEnv(config=env_config, reward_config=reward_config)
    try:
        obs, info = env.reset(seed=seed)
        if not env.observation_space.contains(obs):
            raise RuntimeError("reset observation is outside observation_space")
        if not np.all(np.isfinite(obs)):
            raise RuntimeError("reset observation contains NaN or inf")
        print(
            "Preflight reset: "
            f"initial_goal_distance={info['initial_goal_distance']:.4f} "
            f"position_error={info['position_error']:.4f} "
            f"joint_margin={info['joint_limit_margin_min']:.4f}"
        )

        for idx in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if not np.all(np.isfinite(obs)) or not np.isfinite(reward):
                raise RuntimeError(f"preflight step {idx} produced NaN or inf")
            if info.get("failure_reason") is not None:
                raise RuntimeError(f"preflight step {idx} failed: {info['failure_reason']}")
            print(
                f"Preflight step {idx}: reward={reward:.4f} "
                f"error={info['position_error']:.4f} tau={info['tau_norm']:.3f} "
                f"max_dq={info['max_abs_joint_velocity']:.3f} "
                f"terminated={terminated} truncated={truncated}"
            )
        print("Preflight: passed")
    except Exception as exc:
        print("Preflight: failed")
        print(f"  model_path={getattr(env, 'model_path', env_config.get('model_path'))}")
        print(f"  observation_space={getattr(env, 'observation_space', None)}")
        print(f"  action_space={getattr(env, 'action_space', None)}")
        raise RuntimeError("PPO training preflight failed; not starting training.") from exc
    finally:
        env.close()


def main() -> None:
    args = make_parser().parse_args()
    set_global_seed(args.seed)
    config = load_yaml_config(args.config)
    env_config = config.get("env", {})
    reward_config = config.get("reward", {})
    ppo_config = dict(config.get("ppo", {}))
    policy = ppo_config.pop("policy", "MlpPolicy")

    print_config_summary(env_config, reward_config)
    run_preflight(env_config, reward_config, args.seed)

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise ImportError(
            "Training requires stable-baselines3, torch, gymnasium, pyyaml, and tensorboard. "
            "Install with: pip install gymnasium stable-baselines3 pyyaml tensorboard"
        ) from exc

    run_dir = make_run_dir(args.run_name)
    copied_config = copy_config(args.config, run_dir)
    print(f"Run directory: {run_dir}")
    print(f"Config copy: {copied_config}")

    if args.n_envs <= 1:
        env = Monitor(
            PandaGoalTrackingEnv(config=env_config, reward_config=reward_config),
            filename=str(run_dir / "logs" / "monitor.csv"),
        )
        env.reset(seed=args.seed)
    else:
        vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
        env = vec_cls([_make_env(env_config, reward_config, args.seed, i) for i in range(args.n_envs)])

    eval_env = Monitor(PandaGoalTrackingEnv(config=env_config, reward_config=reward_config))
    eval_env.reset(seed=args.seed + 10_000)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(1_000 // max(args.n_envs, 1), 1),
        save_path=str(run_dir / "checkpoints"),
        name_prefix="ppo_panda_goal_tracking",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "checkpoints" / "best_model"),
        log_path=str(run_dir / "metrics"),
        eval_freq=max(1_000 // max(args.n_envs, 1), 1),
        deterministic=True,
        render=False,
        n_eval_episodes=5,
    )

    if args.resume:
        model = PPO.load(args.resume, env=env, device=args.device, tensorboard_log=str(run_dir / "logs" / "tb"))
    else:
        model = PPO(
            policy,
            env,
            seed=args.seed,
            device=args.device,
            tensorboard_log=str(run_dir / "logs" / "tb"),
            **ppo_config,
        )

    model.learn(total_timesteps=args.total_timesteps, callback=[checkpoint_cb, eval_cb])
    final_model_path = run_dir / "checkpoints" / "final_model.zip"
    model.save(str(final_model_path))

    summary = {
        "run_dir": str(run_dir),
        "final_model": str(final_model_path),
        "best_model_dir": str(run_dir / "checkpoints" / "best_model"),
        "config": str(copied_config),
        "seed": args.seed,
        "total_timesteps": args.total_timesteps,
        "n_envs": args.n_envs,
        "device": args.device,
    }
    save_json(run_dir / "metrics" / "training_summary.json", summary)
    env.close()
    eval_env.close()
    print(f"Saved final model: {final_model_path}")
    print(f"Saved summary: {run_dir / 'metrics' / 'training_summary.json'}")


if __name__ == "__main__":
    main()
