from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


EVAL_FIELDS = [
    "episode",
    "success",
    "final_position_error",
    "episode_return",
    "episode_length",
    "rms_torque",
    "mean_action_delta",
]


def summarize_episodes(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "success_rate": 0.0,
            "mean_final_position_error": 0.0,
            "median_final_position_error": 0.0,
            "mean_episode_return": 0.0,
            "mean_episode_length": 0.0,
            "rms_torque": 0.0,
            "mean_action_delta": 0.0,
        }

    success = np.asarray([float(r["success"]) for r in rows], dtype=float)
    final_error = np.asarray([float(r["final_position_error"]) for r in rows], dtype=float)
    returns = np.asarray([float(r["episode_return"]) for r in rows], dtype=float)
    lengths = np.asarray([float(r["episode_length"]) for r in rows], dtype=float)
    rms_torque = np.asarray([float(r["rms_torque"]) for r in rows], dtype=float)
    action_delta = np.asarray([float(r["mean_action_delta"]) for r in rows], dtype=float)

    return {
        "success_rate": float(np.mean(success)),
        "mean_final_position_error": float(np.mean(final_error)),
        "median_final_position_error": float(np.median(final_error)),
        "mean_episode_return": float(np.mean(returns)),
        "mean_episode_length": float(np.mean(lengths)),
        "rms_torque": float(np.mean(rms_torque)),
        "mean_action_delta": float(np.mean(action_delta)),
    }


def write_eval_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=EVAL_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in EVAL_FIELDS})
