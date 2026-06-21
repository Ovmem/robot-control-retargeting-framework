from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from retargeting.hand_to_panda import (
    HandToPandaRetargeter,
    HandToPandaRetargetingConfig,
)


@dataclass
class ReplayHandObservation:
    """
    Minimal observation object compatible with HandToPandaRetargeter.update().

    它模拟 vision.hand_tracker 输出的对象：
    - detected
    - score
    - landmarks_image
    - landmarks_world
    """

    detected: bool
    score: float
    landmarks_image: np.ndarray
    landmarks_world: np.ndarray | None = None


@dataclass
class RecordedSample:
    frame_id: int
    timestamp: float
    obs: ReplayHandObservation


def parse_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, None)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def parse_int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, None)
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def row_has_full_landmarks(row: dict[str, str], prefix: str) -> bool:
    """
    Check whether the CSV contains full 21 hand landmarks.

    Supported field examples:
    - img_lm_0_x, img_lm_0_y, img_lm_0_z
    - world_lm_0_x, world_lm_0_y, world_lm_0_z
    """
    for i in range(21):
        for axis in ["x", "y", "z"]:
            if f"{prefix}_lm_{i}_{axis}" not in row:
                return False
    return True


def read_landmarks(row: dict[str, str], prefix: str) -> np.ndarray:
    P = np.zeros((21, 3), dtype=float)

    for i in range(21):
        P[i, 0] = parse_float(row, f"{prefix}_lm_{i}_x", 0.0)
        P[i, 1] = parse_float(row, f"{prefix}_lm_{i}_y", 0.0)
        P[i, 2] = parse_float(row, f"{prefix}_lm_{i}_z", 0.0)

    return P


def make_minimal_landmarks_from_logged_features(
    wrist_xyz: tuple[float, float, float],
    pinch_ratio: float,
    palm_span: float = 0.12,
) -> np.ndarray:
    """
    Fallback when old CSV does not store full 21 MediaPipe landmarks.

    注意：
    这不是重新实现 hand-to-panda 映射，只是为了把旧 CSV 里的 wrist/pinch
    重新包装成 HandToPandaRetargeter.update() 能接受的 observation。

    如果你后面在 demo CSV 中记录完整 21 个 landmarks，这个 fallback 会自动不用。
    """
    wx, wy, wz = wrist_xyz
    P = np.zeros((21, 3), dtype=float)

    # Core palm points.
    P[0] = [wx, wy, wz]  # wrist
    P[5] = [wx + 0.045, wy - 0.030, wz]  # index MCP
    P[9] = [wx + 0.015, wy - palm_span, wz]  # middle MCP
    P[17] = [wx - 0.045, wy - 0.030, wz]  # pinky MCP

    # Pinch points. The absolute geometry is approximate, but the ratio is useful
    # for replaying gripper mapping.
    pinch_dist = float(np.clip(pinch_ratio * palm_span, 0.005, 0.20))
    center = np.array([wx + 0.070, wy - 0.035, wz], dtype=float)

    P[4] = center + np.array([+0.5 * pinch_dist, 0.0, 0.0])
    P[8] = center + np.array([-0.5 * pinch_dist, 0.0, 0.0])

    # Fill unused landmarks near wrist to avoid zero-size / invalid palm geometry.
    for i in range(21):
        if i in {0, 4, 5, 8, 9, 17}:
            continue
        P[i] = P[0] + np.array(
            [
                0.005 * (i % 4),
                -0.010 * (1 + i % 5),
                0.0,
            ],
            dtype=float,
        )

    return P


def load_recorded_samples(csv_path: Path) -> list[RecordedSample]:
    samples: list[RecordedSample] = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV: {csv_path}")

        for idx, row in enumerate(reader):
            detected = str(row.get("detected_hand", "1")).strip() in {"1", "true", "True"}

            if not detected:
                continue

            frame_id = parse_int(row, "frame_id", idx)
            timestamp = parse_float(row, "timestamp", float(idx))
            score = parse_float(row, "detection_confidence", 1.0)

            if row_has_full_landmarks(row, "img"):
                landmarks_image = read_landmarks(row, "img")
            elif row_has_full_landmarks(row, "image"):
                landmarks_image = read_landmarks(row, "image")
            else:
                wrist = (
                    parse_float(row, "wrist_x", 0.5),
                    parse_float(row, "wrist_y", 0.5),
                    parse_float(row, "wrist_z", 0.0),
                )
                pinch = parse_float(row, "pinch_ratio", 0.8)
                landmarks_image = make_minimal_landmarks_from_logged_features(wrist, pinch)

            if row_has_full_landmarks(row, "world"):
                landmarks_world = read_landmarks(row, "world")
            else:
                landmarks_world = landmarks_image.copy()

            obs = ReplayHandObservation(
                detected=True,
                score=score,
                landmarks_image=landmarks_image,
                landmarks_world=landmarks_world,
            )

            samples.append(
                RecordedSample(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    obs=obs,
                )
            )

    if len(samples) == 0:
        raise ValueError(
            f"No detected hand samples found in {csv_path}. "
            "Please run demo_hand_retargeting_pd_gc.py first and save a CSV."
        )

    return samples


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def find_latest_hand_retargeting_csv(
    runs_dir: Path,
    filename: str = "hand_retargeting_run.csv",
) -> Path:
    """Find the latest hand_retargeting_run.csv under results/hand_retargeting/runs."""
    if not runs_dir.exists():
        raise FileNotFoundError(
            f"Runs directory not found: {runs_dir}\n"
            "Please run demo_hand_retargeting_pd_gc.py first."
        )

    candidates = list(runs_dir.rglob(filename))

    if not candidates:
        raise FileNotFoundError(
            f"No {filename} found under: {runs_dir}\n"
            "Please run demo_hand_retargeting_pd_gc.py first and make sure CSV logging is enabled."
        )

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def parse_bool_list(text: str) -> list[bool]:
    out: list[bool] = []
    for item in text.split(","):
        s = item.strip().lower()
        if s in {"1", "true", "yes", "y"}:
            out.append(True)
        elif s in {"0", "false", "no", "n"}:
            out.append(False)
        else:
            raise ValueError(f"Cannot parse bool value: {item}")
    return out


def build_config_grid(args: argparse.Namespace) -> list[tuple[str, HandToPandaRetargetingConfig]]:
    scales = parse_float_list(args.position_scales)
    alphas = parse_float_list(args.filter_alphas)
    y_limits = parse_float_list(args.workspace_y_limits)
    z_lowers = parse_float_list(args.workspace_z_lowers)
    z_uppers = parse_float_list(args.workspace_z_uppers)
    orientation_flags = parse_bool_list(args.orientation_flags)
    gripper_flags = parse_bool_list(args.gripper_flags)

    robot_origin = tuple(args.robot_origin)

    configs: list[tuple[str, HandToPandaRetargetingConfig]] = []

    for scale, alpha, y_lim, z_low, z_up, ori_on, grip_on in itertools.product(
        scales,
        alphas,
        y_limits,
        z_lowers,
        z_uppers,
        orientation_flags,
        gripper_flags,
    ):
        name = (
            f"scale{scale:g}_alpha{alpha:g}_"
            f"y{y_lim:g}_z{z_low:g}_{z_up:g}_"
            f"ori{int(ori_on)}_grip{int(grip_on)}"
        )

        cfg = HandToPandaRetargetingConfig(
            robot_origin=robot_origin,
            position_scale_xy=scale,
            vertical_scale_ratio=args.vertical_scale_ratio,
            depth_scale=args.depth_scale,
            enable_depth_mapping=args.enable_depth_mapping,
            min_gripper_width=args.min_gripper_width,
            max_gripper_width=args.max_gripper_width,
            pinch_closed_ratio=args.pinch_closed_ratio,
            pinch_open_ratio=args.pinch_open_ratio,
            filter_alpha=alpha,
            workspace_x_delta=(
                -args.workspace_x_limit,
                args.workspace_x_limit,
            ),
            workspace_y_delta=(-y_lim, y_lim),
            workspace_z_delta=(-z_low, z_up),
            enable_orientation_mapping=ori_on,
            enable_gripper_mapping=grip_on,
            orientation_euler_xyz_deg=tuple(args.orientation_euler_xyz_deg),
        )

        configs.append((name, cfg))

    return configs


def evaluate_config(
    name: str,
    cfg: HandToPandaRetargetingConfig,
    samples: list[RecordedSample],
) -> dict[str, Any]:
    """
    Replay recorded hand observations through the actual HandToPandaRetargeter.

    注意：这里不手写映射逻辑，所有 target pose 都来自：
    retargeting.hand_to_panda.HandToPandaRetargeter.update()
    """
    retargeter = HandToPandaRetargeter(config=cfg)

    positions: list[np.ndarray] = []
    raw_positions: list[np.ndarray] = []
    grippers: list[float] = []
    pinches: list[float] = []
    clipped: list[bool] = []

    valid_count = 0

    for sample in samples:
        target = retargeter.update(sample.obs)

        if not target.valid:
            continue

        valid_count += 1
        positions.append(np.asarray(target.pos, dtype=float).copy())

        if target.raw_pos is not None:
            raw_positions.append(np.asarray(target.raw_pos, dtype=float).copy())
        else:
            raw_positions.append(np.asarray(target.pos, dtype=float).copy())

        grippers.append(float(target.gripper_width))
        pinches.append(float(target.pinch_ratio))
        clipped.append(bool(target.workspace_clipped))

    if valid_count == 0:
        return {
            "name": name,
            **cfg.to_dict(),
            "valid_count": 0,
            "target_smoothness": np.nan,
            "max_target_step": np.nan,
            "target_path_length": np.nan,
            "target_range_yz": np.nan,
            "workspace_clip_rate": 1.0,
            "gripper_smoothness": np.nan,
            "gripper_range": np.nan,
            "score": np.inf,
        }

    P = np.vstack(positions)
    G = np.asarray(grippers, dtype=float)

    if len(P) > 1:
        steps = np.linalg.norm(np.diff(P, axis=0), axis=1)
        target_smoothness = float(np.nanmean(steps))
        max_target_step = float(np.nanmax(steps))
        target_path_length = float(np.nansum(steps))
    else:
        target_smoothness = 0.0
        max_target_step = 0.0
        target_path_length = 0.0

    y_range = float(np.nanmax(P[:, 1]) - np.nanmin(P[:, 1]))
    z_range = float(np.nanmax(P[:, 2]) - np.nanmin(P[:, 2]))
    target_range_yz = float(np.sqrt(y_range**2 + z_range**2))

    if len(G) > 1:
        gripper_diffs = np.abs(np.diff(G))
        gripper_smoothness = float(np.nanmean(gripper_diffs))
    else:
        gripper_smoothness = 0.0

    gripper_range = float(np.nanmax(G) - np.nanmin(G))
    workspace_clip_rate = float(np.mean(clipped)) if clipped else 1.0

    # Score design:
    # - Prefer low clipping.
    # - Prefer smooth target.
    # - Avoid choosing a degenerate config that barely moves.
    min_useful_range = 0.04
    range_penalty = max(0.0, min_useful_range - target_range_yz)

    score = (
        1.00 * target_smoothness
        + 0.50 * workspace_clip_rate
        + 0.10 * gripper_smoothness
        + 2.00 * range_penalty
    )

    return {
        "name": name,
        **cfg.to_dict(),
        "valid_count": valid_count,
        "target_smoothness": target_smoothness,
        "max_target_step": max_target_step,
        "target_path_length": target_path_length,
        "target_range_yz": target_range_yz,
        "workspace_clip_rate": workspace_clip_rate,
        "gripper_smoothness": gripper_smoothness,
        "gripper_range": gripper_range,
        "score": float(score),
    }


def stringify_for_csv(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist())
    if isinstance(value, (tuple, list)):
        return json.dumps(list(value))
    if isinstance(value, bool):
        return int(value)
    return value


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("No rows to write.")

    fieldnames: list[str] = []
    seen: set[str] = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for row in rows:
            writer.writerow({k: stringify_for_csv(v) for k, v in row.items()})


def write_key_value_csv(path: Path, data: dict[str, Any], key_name: str = "param") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([key_name, "value"])

        for k, v in data.items():
            writer.writerow([k, stringify_for_csv(v)])


def extract_config_from_result(best: dict[str, Any]) -> HandToPandaRetargetingConfig:
    config_keys = set(HandToPandaRetargetingConfig.__dataclass_fields__.keys())
    cfg_data = {k: v for k, v in best.items() if k in config_keys}
    return HandToPandaRetargetingConfig.from_dict(cfg_data)


def extract_metrics_from_result(best: dict[str, Any]) -> dict[str, Any]:
    metric_keys = [
        "name",
        "valid_count",
        "target_smoothness",
        "max_target_step",
        "target_path_length",
        "target_range_yz",
        "workspace_clip_rate",
        "gripper_smoothness",
        "gripper_range",
        "score",
    ]
    return {k: best.get(k) for k in metric_keys}


def plot_metric_bar(
    rows: list[dict[str, Any]],
    metric: str,
    output_path: Path,
    ylabel: str,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    names = [str(r["name"]) for r in rows]
    values = [float(r.get(metric, np.nan)) for r in rows]

    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 0.35), 4.5))
    ax.bar(names, values)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=65)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay recorded hand retargeting data through "
            "retargeting.hand_to_panda.HandToPandaRetargeter "
            "with different mapping configs."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help=(
            "Path to hand_retargeting_run.csv generated by demo_hand_retargeting_pd_gc.py. "
            "If omitted, the latest CSV under results/hand_retargeting/runs will be used."
        ),
    )

    parser.add_argument(
        "--runs-dir",
        type=str,
        default="results/hand_retargeting/runs",
        help="Directory used to search the latest hand_retargeting_run.csv when --input is omitted.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Default: <run_dir>/param_sweep",
    )

    parser.add_argument(
        "--robot-origin",
        type=float,
        nargs=3,
        default=[0.45, 0.0, 0.45],
        help="Fallback robot origin used during replay.",
    )

    parser.add_argument(
        "--position-scales",
        type=str,
        default="1.4,2.2,3.0,3.5",
        help="Comma-separated position_scale_xy candidates.",
    )
    parser.add_argument(
        "--filter-alphas",
        type=str,
        default="0.08,0.18,0.40,0.65",
        help="Comma-separated filter_alpha candidates.",
    )
    parser.add_argument(
        "--workspace-y-limits",
        type=str,
        default="0.15,0.25",
        help="Comma-separated symmetric y workspace half-widths.",
    )
    parser.add_argument(
        "--workspace-z-lowers",
        type=str,
        default="0.08,0.12",
        help="Comma-separated lower z workspace deltas.",
    )
    parser.add_argument(
        "--workspace-z-uppers",
        type=str,
        default="0.10,0.14",
        help="Comma-separated upper z workspace deltas.",
    )
    parser.add_argument(
        "--workspace-x-limit",
        type=float,
        default=0.02,
        help="Symmetric x workspace half-width when depth mapping is disabled.",
    )

    parser.add_argument(
        "--orientation-flags",
        type=str,
        default="true,false",
        help="Comma-separated bools for enable_orientation_mapping.",
    )
    parser.add_argument(
        "--gripper-flags",
        type=str,
        default="true,false",
        help="Comma-separated bools for enable_gripper_mapping.",
    )

    parser.add_argument(
        "--vertical-scale-ratio",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--enable-depth-mapping",
        action="store_true",
    )

    parser.add_argument(
        "--min-gripper-width",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--max-gripper-width",
        type=float,
        default=0.04,
    )
    parser.add_argument(
        "--pinch-closed-ratio",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--pinch-open-ratio",
        type=float,
        default=1.20,
    )

    parser.add_argument(
        "--orientation-euler-xyz-deg",
        type=float,
        nargs=3,
        default=[180.0, 0.0, 90.0],
    )

    return parser


def main() -> None:
    args = make_parser().parse_args()
    
    if args.input is None:
        runs_dir = Path(args.runs_dir)
        if not runs_dir.is_absolute():
            runs_dir = PROJECT_ROOT / runs_dir
        csv_path = find_latest_hand_retargeting_csv(runs_dir)
        print(f"No --input provided. Using latest CSV: {csv_path}")
    
    else:
        csv_path = Path(args.input)
        if not csv_path.is_absolute():
            csv_path = PROJECT_ROOT / csv_path

        if not csv_path.exists():
            raise FileNotFoundError(f"Input CSV not found: {csv_path}")


    run_dir = csv_path.parents[1]

    if args.output_dir is None:
        sweep_dir = run_dir / "param_sweep"
    else:
        sweep_dir = Path(args.output_dir)

    raw_dir = sweep_dir / "raw"
    figures_dir = sweep_dir / "figures"
    metrics_dir = sweep_dir / "metrics"

    raw_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading recorded hand data: {csv_path}")
    samples = load_recorded_samples(csv_path)
    print(f"Loaded {len(samples)} detected samples.")

    configs = build_config_grid(args)
    print(f"Running {len(configs)} retargeting configs...")

    results: list[dict[str, Any]] = []

    for name, cfg in configs:
        row = evaluate_config(name, cfg, samples)
        results.append(row)
        print(
            f"{name:55s} "
            f"score={row['score']:.6f} "
            f"smooth={row['target_smoothness']:.6f} "
            f"clip={row['workspace_clip_rate']:.3f} "
            f"range={row['target_range_yz']:.3f}"
        )

    results_sorted = sorted(results, key=lambda r: float(r["score"]))
    best = results_sorted[0]

    # 1. Full sweep result: params + metrics.
    full_csv = raw_dir / "param_sweep_results.csv"
    write_rows_csv(full_csv, results_sorted)

    # 2. Clean best params only.
    best_cfg = extract_config_from_result(best)
    best_params_csv = metrics_dir / "best_params.csv"
    best_params_json = metrics_dir / "best_params.json"
    best_cfg.to_csv(best_params_csv)
    best_cfg.to_json(best_params_json)

    # 3. Best metrics only.
    best_metrics = extract_metrics_from_result(best)
    best_metrics_csv = metrics_dir / "best_metrics.csv"
    write_key_value_csv(best_metrics_csv, best_metrics, key_name="metric")

    # 4. Plots.
    plot_metric_bar(
        results_sorted,
        "score",
        figures_dir / "param_sweep_score.png",
        ylabel="Score",
        title="Hand Retargeting Parameter Sweep: Score",
    )
    plot_metric_bar(
        results_sorted,
        "target_smoothness",
        figures_dir / "param_sweep_target_smoothness.png",
        ylabel="Mean target step [m]",
        title="Hand Retargeting Parameter Sweep: Target Smoothness",
    )
    plot_metric_bar(
        results_sorted,
        "workspace_clip_rate",
        figures_dir / "param_sweep_workspace_clip_rate.png",
        ylabel="Workspace clip rate",
        title="Hand Retargeting Parameter Sweep: Workspace Clip Rate",
    )
    plot_metric_bar(
        results_sorted,
        "target_range_yz",
        figures_dir / "param_sweep_target_range_yz.png",
        ylabel="Target YZ range [m]",
        title="Hand Retargeting Parameter Sweep: Target Motion Range",
    )

    print("\nSweep complete.")
    print(f"Full results:   {full_csv}")
    print(f"Best params:    {best_params_csv}")
    print(f"Best params:    {best_params_json}")
    print(f"Best metrics:   {best_metrics_csv}")
    print(f"Best config:    {best['name']}")
    print(f"Best score:     {best['score']:.6f}")


if __name__ == "__main__":
    main()
