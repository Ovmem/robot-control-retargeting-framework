from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from learning.envs import PandaGoalTrackingEnv  # noqa: E402
from learning.utils.metrics import summarize_episodes, write_eval_csv  # noqa: E402
from learning.utils.run_utils import load_yaml_config, resolve_project_path, save_json  # noqa: E402


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a Panda goal-tracking PPO policy.")
    parser.add_argument("--model", required=True, help="Path to a Stable-Baselines3 PPO model zip.")
    parser.add_argument("--config", default="learning/configs/panda_ppo_default.yaml")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    config = load_yaml_config(args.config)
    env_config = config.get("env", {})
    reward_config = config.get("reward", {})
    render_mode = "rgb_array" if args.record_video else ("human" if args.render else None)

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise ImportError(
            "Evaluation requires stable-baselines3, torch, gymnasium, and pyyaml. "
            "Install with: pip install gymnasium stable-baselines3 pyyaml tensorboard"
        ) from exc

    env = PandaGoalTrackingEnv(
        config=env_config,
        reward_config=reward_config,
        render_mode=render_mode,
    )
    model_path = resolve_project_path(args.model)
    model = PPO.load(str(model_path), device="auto")

    if args.output_dir:
        output_dir = resolve_project_path(args.output_dir)
    else:
        run_dir = model_path.parent.parent
        if model_path.parent.name == "best_model":
            run_dir = model_path.parents[2]
        output_dir = run_dir / "metrics" / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = output_dir / "videos"
    if args.record_video:
        video_dir.mkdir(parents=True, exist_ok=True)

    video_writer = None
    if args.record_video:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - optional video dependency guard
            raise ImportError("Video recording requires opencv-python.") from exc

    rows = []
    for episode in range(args.episodes):
        obs, info = env.reset(seed=args.seed + episode)
        done = False
        episode_return = 0.0
        episode_length = 0
        tau_sq = []
        action_delta = []
        prev_action = np.zeros(env.action_space.shape, dtype=np.float32)
        video_writer = None

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            frame = env.render() if (args.render or args.record_video) else None
            if args.record_video and frame is not None:
                if video_writer is None:
                    height, width = frame.shape[:2]
                    video_path = video_dir / f"episode_{episode:03d}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(str(video_path), fourcc, 50.0, (width, height))
                video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            applied = np.asarray(info.get("applied_action", action), dtype=np.float32)
            action_delta.append(float(np.linalg.norm(applied - prev_action)))
            prev_action = applied.copy()
            tau_sq.append(float(info["torque_norm"]) ** 2)
            episode_return += float(reward)
            episode_length += 1
            done = bool(terminated or truncated)

        if video_writer is not None:
            video_writer.release()

        rows.append(
            {
                "episode": episode,
                "success": int(info.get("is_success", False)),
                "final_position_error": float(info["position_error"]),
                "episode_return": episode_return,
                "episode_length": episode_length,
                "rms_torque": float(np.sqrt(np.mean(tau_sq))) if tau_sq else 0.0,
                "mean_action_delta": float(np.mean(action_delta)) if action_delta else 0.0,
            }
        )

    summary = summarize_episodes(rows)
    csv_path = output_dir / "evaluation_episodes.csv"
    json_path = output_dir / "evaluation_summary.json"
    write_eval_csv(csv_path, rows)
    save_json(json_path, {"model": str(model_path), "episodes": args.episodes, **summary})
    env.close()

    print(f"Saved episode CSV: {csv_path}")
    print(f"Saved summary JSON: {json_path}")
    print(summary)


if __name__ == "__main__":
    main()
