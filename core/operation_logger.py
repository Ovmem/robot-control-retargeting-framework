"""Operation data recording interface.

Records robot-operation samples for downstream policy-verification
or imitation-learning data-format prototyping.

This module does NOT train models or interface with real hardware.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass
class OperationSample:
    """A single operation-data sample.

    Fields prefixed with *target_* describe the commanded target;
    *current_* fields (optional) carry the observed state.
    *action_pos_delta* is the position increment applied at this step.
    """

    step: int
    time: float
    target_pos: np.ndarray          # (3,)
    target_rot: Optional[np.ndarray] = None   # (3, 3) or None
    current_pos: Optional[np.ndarray] = None  # (3,) or None
    current_qpos: Optional[np.ndarray] = None # (7,) or None
    action_pos_delta: Optional[np.ndarray] = None  # (3,) or None
    gripper_width: float = 0.0
    valid: bool = True
    source: str = "mock_hand"


def _finite(arr, name):
    """Raise if *arr* is not None and contains NaN or inf."""
    if arr is not None and not np.all(np.isfinite(arr)):
        raise ValueError(f"non-finite value in {name}")


class OperationLogger:
    """Append-only logger for operation samples.

    Usage::

        logger = OperationLogger()
        logger.append(sample)
        logger.save_csv("traj.csv")
        logger.save_npz("traj.npz")
        data = OperationLogger.load_npz("traj.npz")
    """

    CSV_FIELDS = [
        "step", "time",
        "target_pos_x", "target_pos_y", "target_pos_z",
        "current_pos_x", "current_pos_y", "current_pos_z",
        "action_dx", "action_dy", "action_dz",
        "gripper_width", "valid", "source",
    ]

    def __init__(self):
        self.samples: List[OperationSample] = []

    def append(self, sample: OperationSample):
        _finite(sample.target_pos, "target_pos")
        _finite(sample.current_pos, "current_pos")
        _finite(sample.action_pos_delta, "action_pos_delta")
        self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _v3(arr):
        """Return a 3-element list from arr or [None]*3."""
        if arr is None:
            return [None, None, None]
        return [float(arr[0]), float(arr[1]), float(arr[2])]

    def save_csv(self, path):
        """Save all samples to a CSV file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=self.CSV_FIELDS)
            w.writeheader()
            for s in self.samples:
                tp = self._v3(s.target_pos)
                cp = self._v3(s.current_pos)
                ad = self._v3(s.action_pos_delta)
                w.writerow({
                    "step": s.step, "time": s.time,
                    "target_pos_x": tp[0], "target_pos_y": tp[1], "target_pos_z": tp[2],
                    "current_pos_x": cp[0], "current_pos_y": cp[1], "current_pos_z": cp[2],
                    "action_dx": ad[0], "action_dy": ad[1], "action_dz": ad[2],
                    "gripper_width": s.gripper_width,
                    "valid": int(s.valid), "source": s.source,
                })

    def save_npz(self, path):
        """Save all samples to a compressed .npz archive."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        n = len(self.samples)

        target_pos = np.full((n, 3), np.nan, dtype=np.float64)
        current_pos = np.full((n, 3), np.nan, dtype=np.float64)
        action_delta = np.full((n, 3), np.nan, dtype=np.float64)
        step_a = np.zeros(n, dtype=np.int64)
        t = np.zeros(n, dtype=np.float64)
        gripper = np.zeros(n, dtype=np.float64)
        valid = np.zeros(n, dtype=bool)
        source = []

        for i, s in enumerate(self.samples):
            step_a[i] = s.step
            t[i] = s.time
            if s.target_pos is not None:
                target_pos[i] = s.target_pos
            if s.current_pos is not None:
                current_pos[i] = s.current_pos
            if s.action_pos_delta is not None:
                action_delta[i] = s.action_pos_delta
            gripper[i] = s.gripper_width
            valid[i] = s.valid
            source.append(s.source)

        np.savez_compressed(
            path,
            step=step_a, time=t,
            target_pos=target_pos,
            current_pos=current_pos,
            action_delta=action_delta,
            gripper_width=gripper,
            valid=valid,
            source=np.array(source, dtype=object),
        )

    @staticmethod
    def load_npz(path):
        """Load arrays from a .npz file written by save_npz.

        Returns a dict with keys: step, time, target_pos, current_pos,
        action_delta, gripper_width, valid, source.
        """
        data = np.load(path, allow_pickle=True)
        return {k: data[k] for k in data}
