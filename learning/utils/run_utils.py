from __future__ import annotations

import json
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def require_yaml():
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise ImportError("Install PyYAML with: pip install pyyaml") from exc
    return yaml


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    yaml = require_yaml()
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with open(config_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def make_run_dir(run_name: str, root: str | Path = "results/rl_tracking/runs") -> Path:
    root_path = Path(root)
    if not root_path.is_absolute():
        root_path = PROJECT_ROOT / root_path
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in run_name).strip("_")
    safe_name = safe_name or "run"
    run_dir = root_path / f"{stamp}_{safe_name}"
    for child in ["checkpoints", "logs", "metrics", "config", "videos"]:
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    return run_dir


def copy_config(config_path: str | Path, run_dir: Path) -> Path:
    src = Path(config_path)
    if not src.is_absolute():
        src = PROJECT_ROOT / src
    dst = run_dir / "config" / src.name
    shutil.copy2(src, dst)
    return dst


def resolve_project_path(path: str | Path) -> Path:
    out = Path(path)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    return out
