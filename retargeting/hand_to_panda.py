from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy.spatial.transform import Rotation


def normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Return a unit vector. If the norm is too small, return zeros."""
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def _as_float_tuple(value: Any, length: int | None = None) -> tuple[float, ...]:
    """Convert list / tuple / ndarray / string to a tuple of floats."""
    if isinstance(value, str):
        s = value.strip()
        try:
            value = json.loads(s)
        except json.JSONDecodeError:
            value = [x.strip() for x in s.replace(";", ",").split(",") if x.strip()]

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, (list, tuple)):
        out = tuple(float(x) for x in value)
    else:
        out = (float(value),)

    if length is not None and len(out) != length:
        raise ValueError(f"Expected length {length}, got {len(out)}: {out}")

    return out


def _parse_value(value: str) -> Any:
    """Parse a CSV config value."""
    s = str(value).strip()

    if s.lower() in {"true", "false"}:
        return s.lower() == "true"

    if s.lower() in {"none", "null"}:
        return None

    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
        return json.loads(s)

    try:
        if any(c in s for c in [".", "e", "E"]):
            return float(s)
        return int(s)
    except ValueError:
        return s


@dataclass
class PandaTarget:
    """Retargeted target for Panda end-effector."""

    pos: np.ndarray
    rot: np.ndarray
    gripper_width: float
    pinch_ratio: float
    valid: bool = True

    # Extra debug fields. Existing demos can ignore them safely.
    raw_pos: Optional[np.ndarray] = None
    workspace_clipped: bool = False


@dataclass
class HandToPandaRetargetingConfig:
    """
    Config for hand landmarks -> Panda end-effector target mapping.

    主要参数解释：
    - position_scale_xy：图像中手腕平移到机器人 y/z 平移的缩放。
    - vertical_scale_ratio：图像 y 方向映射到机器人 z 方向时的额外缩放。
    - filter_alpha：低通滤波系数。当前实现中 alpha 越大，越偏向上一帧，运动越平滑但响应越慢。
    - workspace_*_delta：相对 robot_origin 的工作空间裁剪范围。
    - enable_orientation_mapping：是否使用手掌坐标系映射末端姿态。
    - enable_gripper_mapping：是否使用捏合比例映射夹爪宽度。
    """

    robot_origin: tuple[float, float, float] = (0.45, 0.0, 0.45)

    position_scale_xy: float = 2.2
    vertical_scale_ratio: float = 0.8

    depth_scale: float = 0.0
    enable_depth_mapping: bool = False

    min_gripper_width: float = 0.0
    max_gripper_width: float = 0.04
    pinch_closed_ratio: float = 0.35
    pinch_open_ratio: float = 1.20

    filter_alpha: float = 0.18

    workspace_x_delta: tuple[float, float] = (-0.02, 0.02)
    workspace_y_delta: tuple[float, float] = (-0.25, 0.25)
    workspace_z_delta: tuple[float, float] = (-0.12, 0.14)

    enable_orientation_mapping: bool = True
    enable_gripper_mapping: bool = True
    orientation_euler_xyz_deg: tuple[float, float, float] = (180.0, 0.0, 90.0)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HandToPandaRetargetingConfig":
        valid_keys = set(cls.__dataclass_fields__.keys())
        cleaned: dict[str, Any] = {}

        for k, v in data.items():
            if k not in valid_keys:
                continue

            if k in {
                "robot_origin",
                "workspace_x_delta",
                "workspace_y_delta",
                "workspace_z_delta",
                "orientation_euler_xyz_deg",
            }:
                expected_len = 3 if k in {"robot_origin", "orientation_euler_xyz_deg"} else 2
                cleaned[k] = _as_float_tuple(v, expected_len)
            else:
                cleaned[k] = v

        return cls(**cleaned)

    @classmethod
    def from_json(cls, path: str | Path) -> "HandToPandaRetargetingConfig":
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_csv(cls, path: str | Path) -> "HandToPandaRetargetingConfig":
        """
        Load config from a param,value CSV.

        Expected format:
        param,value
        position_scale_xy,2.2
        filter_alpha,0.18
        workspace_y_delta,"[-0.25, 0.25]"
        """
        path = Path(path)
        data: dict[str, Any] = {}

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "param" not in reader.fieldnames or "value" not in reader.fieldnames:
                raise ValueError(
                    f"{path} must be a param,value CSV. Got fields: {reader.fieldnames}"
                )

            for row in reader:
                key = row["param"].strip()
                value = _parse_value(row["value"])
                data[key] = value

        return cls.from_dict(data)

    @classmethod
    def from_file(cls, path: str | Path) -> "HandToPandaRetargetingConfig":
        path = Path(path)
        if path.suffix.lower() == ".json":
            return cls.from_json(path)
        if path.suffix.lower() == ".csv":
            return cls.from_csv(path)
        raise ValueError(f"Unsupported config format: {path}")

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def to_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["param", "value"])
            for k, v in self.to_dict().items():
                if isinstance(v, (list, tuple)):
                    v = json.dumps(v)
                writer.writerow([k, v])


class LowPassFilter:
    """
    Exponential moving-average filter.

    y_t = alpha * y_{t-1} + (1 - alpha) * x_t

    当前实现里：
    - alpha 越大：越平滑，但响应越慢。
    - alpha 越小：响应越快，但更容易抖。
    """

    def __init__(self, alpha: float = 0.55):
        self.alpha = float(np.clip(alpha, 0.0, 0.999))
        self.y: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.y = None

    def update(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)

        if self.y is None:
            self.y = x.copy()
        else:
            self.y = self.alpha * self.y + (1.0 - self.alpha) * x

        return self.y.copy()


def project_to_rotation_matrix(R: np.ndarray) -> np.ndarray:
    """
    Project a near-rotation matrix back to SO(3).

    低通滤波直接作用在矩阵元素上会破坏正交性，所以需要投影回合法旋转矩阵。
    """
    R = np.asarray(R, dtype=float)

    if R.shape != (3, 3) or not np.isfinite(R).all():
        return np.eye(3)

    U, _, Vt = np.linalg.svd(R)
    R_proj = U @ Vt

    if np.linalg.det(R_proj) < 0:
        U[:, -1] *= -1.0
        R_proj = U @ Vt

    return R_proj


class HandToPandaRetargeter:
    """
    Hand landmarks -> Panda end-effector target.

    输入：
    - obs.landmarks_image: MediaPipe image landmarks, shape (21, 3)
    - obs.landmarks_world: MediaPipe world landmarks, shape (21, 3), optional

    输出：
    - PandaTarget.pos: Panda 末端目标位置
    - PandaTarget.rot: Panda 末端目标姿态
    - PandaTarget.gripper_width: 夹爪宽度指令
    """

    def __init__(
        self,
        config: Optional[HandToPandaRetargetingConfig] = None,
        robot_origin: Optional[np.ndarray] = None,
        position_scale_xy: Optional[float] = None,
        depth_scale: Optional[float] = None,
        enable_depth_mapping: Optional[bool] = None,
        min_gripper_width: Optional[float] = None,
        max_gripper_width: Optional[float] = None,
        filter_alpha: Optional[float] = None,
        position_scale: Optional[float] = None,
        workspace_x_delta: Optional[tuple[float, float]] = None,
        workspace_y_delta: Optional[tuple[float, float]] = None,
        workspace_z_delta: Optional[tuple[float, float]] = None,
        enable_orientation_mapping: Optional[bool] = None,
        enable_gripper_mapping: Optional[bool] = None,
    ):
        # Keep backward compatibility with previous demo code.
        cfg = config if config is not None else HandToPandaRetargetingConfig()
        cfg = HandToPandaRetargetingConfig.from_dict(cfg.to_dict())

        if robot_origin is not None:
            cfg.robot_origin = tuple(float(x) for x in np.asarray(robot_origin).reshape(3))

        if position_scale is not None:
            cfg.position_scale_xy = float(position_scale)

        if position_scale_xy is not None:
            cfg.position_scale_xy = float(position_scale_xy)

        if depth_scale is not None:
            cfg.depth_scale = float(depth_scale)

        if enable_depth_mapping is not None:
            cfg.enable_depth_mapping = bool(enable_depth_mapping)

        if min_gripper_width is not None:
            cfg.min_gripper_width = float(min_gripper_width)

        if max_gripper_width is not None:
            cfg.max_gripper_width = float(max_gripper_width)

        if filter_alpha is not None:
            cfg.filter_alpha = float(filter_alpha)

        if workspace_x_delta is not None:
            cfg.workspace_x_delta = _as_float_tuple(workspace_x_delta, 2)

        if workspace_y_delta is not None:
            cfg.workspace_y_delta = _as_float_tuple(workspace_y_delta, 2)

        if workspace_z_delta is not None:
            cfg.workspace_z_delta = _as_float_tuple(workspace_z_delta, 2)

        if enable_orientation_mapping is not None:
            cfg.enable_orientation_mapping = bool(enable_orientation_mapping)

        if enable_gripper_mapping is not None:
            cfg.enable_gripper_mapping = bool(enable_gripper_mapping)

        self.config = cfg
        self.robot_origin = np.asarray(cfg.robot_origin, dtype=float)

        self.base_wrist_img: Optional[np.ndarray] = None
        self.base_palm_size: Optional[float] = None

        self.pos_filter = LowPassFilter(alpha=cfg.filter_alpha)
        self.rot_filter = LowPassFilter(alpha=cfg.filter_alpha)

        self.last_raw_pos: Optional[np.ndarray] = None
        self.last_clipped_pos: Optional[np.ndarray] = None
        self.last_workspace_clipped: bool = False

    def reset_origin(self) -> None:
        self.base_wrist_img = None
        self.base_palm_size = None
        self.pos_filter.reset()
        self.rot_filter.reset()
        self.last_raw_pos = None
        self.last_clipped_pos = None
        self.last_workspace_clipped = False

    @staticmethod
    def _valid_landmarks(P: Any) -> bool:
        if P is None:
            return False
        P = np.asarray(P)
        return P.shape[0] >= 21 and P.shape[1] >= 3 and np.isfinite(P).all()

    @staticmethod
    def choose_landmarks(obs: Any) -> np.ndarray:
        """
        Prefer world landmarks for orientation and pinch if available.
        Fall back to image landmarks.
        """
        P_world = getattr(obs, "landmarks_world", None)
        P_img = getattr(obs, "landmarks_image", None)

        if HandToPandaRetargeter._valid_landmarks(P_world):
            return np.asarray(P_world, dtype=float).copy()

        if HandToPandaRetargeter._valid_landmarks(P_img):
            return np.asarray(P_img, dtype=float).copy()

        raise ValueError("No valid hand landmarks found in observation.")

    @staticmethod
    def palm_frame(P: np.ndarray) -> np.ndarray:
        """
        Build a palm frame from MediaPipe 21 hand landmarks.

        0  : wrist
        5  : index MCP
        9  : middle MCP
        17 : pinky MCP
        """
        P = np.asarray(P, dtype=float)

        wrist = P[0]
        index_mcp = P[5]
        middle_mcp = P[9]
        pinky_mcp = P[17]

        x_axis = normalize(index_mcp - pinky_mcp)
        y_raw = normalize(middle_mcp - wrist)
        z_axis = normalize(np.cross(x_axis, y_raw))
        y_axis = normalize(np.cross(z_axis, x_axis))

        R_hand = np.column_stack([x_axis, y_axis, z_axis])

        if not np.isfinite(R_hand).all() or abs(np.linalg.det(R_hand)) < 1e-6:
            return np.eye(3)

        if np.linalg.det(R_hand) < 0:
            R_hand[:, 2] *= -1.0

        return project_to_rotation_matrix(R_hand)

    @staticmethod
    def pinch_ratio(P: np.ndarray) -> float:
        """
        thumb tip 到 index tip 的距离 / 手掌尺度。
        数值越小，说明越接近捏合。
        """
        P = np.asarray(P, dtype=float)

        thumb_tip = P[4]
        index_tip = P[8]
        wrist = P[0]
        middle_mcp = P[9]

        pinch = np.linalg.norm(thumb_tip - index_tip)
        palm = np.linalg.norm(middle_mcp - wrist) + 1e-9

        return float(pinch / palm)

    def _clip_position(self, pos: np.ndarray) -> tuple[np.ndarray, bool]:
        cfg = self.config
        clipped = pos.copy()

        x_min = self.robot_origin[0] + cfg.workspace_x_delta[0]
        x_max = self.robot_origin[0] + cfg.workspace_x_delta[1]
        y_min = self.robot_origin[1] + cfg.workspace_y_delta[0]
        y_max = self.robot_origin[1] + cfg.workspace_y_delta[1]
        z_min = self.robot_origin[2] + cfg.workspace_z_delta[0]
        z_max = self.robot_origin[2] + cfg.workspace_z_delta[1]

        clipped[0] = np.clip(clipped[0], x_min, x_max)
        clipped[1] = np.clip(clipped[1], y_min, y_max)
        clipped[2] = np.clip(clipped[2], z_min, z_max)

        was_clipped = bool(np.linalg.norm(clipped - pos) > 1e-9)
        return clipped, was_clipped

    def map_position_from_image(self, P_img: np.ndarray) -> np.ndarray:
        """
        Use image landmarks for hand translation mapping.

        P_img[:, 0]：图像 x，向右增大 -> robot y
        P_img[:, 1]：图像 y，向下增大 -> robot z 取反
        """
        cfg = self.config
        P_img = np.asarray(P_img, dtype=float)

        wrist = P_img[0]
        middle_mcp = P_img[9]
        palm_size = np.linalg.norm(middle_mcp[:2] - wrist[:2])

        if self.base_wrist_img is None:
            self.base_wrist_img = wrist.copy()
            self.base_palm_size = palm_size

        delta = wrist - self.base_wrist_img
        delta_palm = palm_size - float(self.base_palm_size)

        dx_img = delta[0]
        dy_img = delta[1]

        if cfg.enable_depth_mapping:
            depth_dx = cfg.depth_scale * delta_palm
        else:
            depth_dx = 0.0

        robot_delta = np.array(
            [
                depth_dx,
                cfg.position_scale_xy * dx_img,
                -cfg.vertical_scale_ratio * cfg.position_scale_xy * dy_img,
            ],
            dtype=float,
        )

        raw_pos = self.robot_origin + robot_delta
        clipped_pos, was_clipped = self._clip_position(raw_pos)

        self.last_raw_pos = raw_pos.copy()
        self.last_clipped_pos = clipped_pos.copy()
        self.last_workspace_clipped = was_clipped

        return clipped_pos

    def map_orientation(self, R_hand: np.ndarray) -> np.ndarray:
        """Map palm frame to Panda end-effector frame."""
        if not self.config.enable_orientation_mapping:
            return np.eye(3)

        R_robot_from_hand = Rotation.from_euler(
            "xyz",
            self.config.orientation_euler_xyz_deg,
            degrees=True,
        ).as_matrix()

        return project_to_rotation_matrix(R_robot_from_hand @ R_hand)

    def map_gripper(self, pinch_ratio: float) -> float:
        """Map pinch ratio to gripper width."""
        cfg = self.config

        if not cfg.enable_gripper_mapping:
            return float(cfg.max_gripper_width)

        denom = cfg.pinch_open_ratio - cfg.pinch_closed_ratio
        if abs(denom) < 1e-9:
            s = 1.0
        else:
            s = (pinch_ratio - cfg.pinch_closed_ratio) / denom

        s = float(np.clip(s, 0.0, 1.0))
        return float(cfg.min_gripper_width + s * (cfg.max_gripper_width - cfg.min_gripper_width))

    def update(self, obs: Any) -> PandaTarget:
        """Compute Panda target from one hand observation."""
        detected = getattr(obs, "detected", True)
        if detected is False:
            return PandaTarget(
                pos=self.robot_origin.copy(),
                rot=np.eye(3),
                gripper_width=self.config.max_gripper_width,
                pinch_ratio=np.nan,
                valid=False,
            )

        P_img = getattr(obs, "landmarks_image", None)
        if not self._valid_landmarks(P_img):
            return PandaTarget(
                pos=self.robot_origin.copy(),
                rot=np.eye(3),
                gripper_width=self.config.max_gripper_width,
                pinch_ratio=np.nan,
                valid=False,
            )

        P_img = np.asarray(P_img, dtype=float)
        P_for_frame = self.choose_landmarks(obs)

        raw_or_clipped_pos = self.map_position_from_image(P_img)
        filtered_pos = self.pos_filter.update(raw_or_clipped_pos)

        R_hand = self.palm_frame(P_for_frame)
        raw_rot = self.map_orientation(R_hand)
        filtered_rot = self.rot_filter.update(raw_rot)
        filtered_rot = project_to_rotation_matrix(filtered_rot)

        pinch = self.pinch_ratio(P_for_frame)
        gripper_width = self.map_gripper(pinch)

        return PandaTarget(
            pos=filtered_pos,
            rot=filtered_rot,
            gripper_width=gripper_width,
            pinch_ratio=pinch,
            valid=True,
            raw_pos=self.last_raw_pos.copy() if self.last_raw_pos is not None else None,
            workspace_clipped=self.last_workspace_clipped,
        )

