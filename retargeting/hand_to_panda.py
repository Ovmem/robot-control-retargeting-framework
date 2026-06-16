# retargeting/hand_to_panda.py

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation


def normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


@dataclass
class PandaTarget:
    pos: np.ndarray
    rot: np.ndarray
    gripper_width: float
    pinch_ratio: float
    valid: bool = True


class LowPassFilter:
    """Exponential moving-average filter for smoothing."""
    def __init__(self, alpha: float = 0.55):
        self.alpha = alpha
        self.y = None

    def reset(self):
        self.y = None

    def update(self, x: np.ndarray) -> np.ndarray:
        """Apply one step of exponential smoothing."""
        if self.y is None:
            self.y = x.copy()
        else:
            self.y = self.alpha * self.y + (1.0 - self.alpha) * x
        return self.y.copy()


class HandToPandaRetargeter:
    """
    Hand landmarks -> Panda end-effector target.

    设计逻辑：
    1. 平动：使用 image landmarks 的 wrist 点，更适合感知手在画面中的上下左右移动。
    2. 姿态：优先使用 world landmarks，根据手掌坐标系计算目标姿态。
    3. 夹爪：使用 thumb tip 和 index tip 的距离估计 pinch ratio。
    """

    def __init__(
        self,
        robot_origin: np.ndarray = np.array([0.45, 0.0, 0.45]),
        position_scale_xy: float = 1.4,
        depth_scale: float = 0.8,
        min_gripper_width: float = 0.0,
        max_gripper_width: float = 0.04,
        filter_alpha: float = 0.55,
        position_scale: Optional[float] = None,
    ):
        """
        position_scale 是为了兼容你之前 demo 里的旧参数名。
        如果 demo 里还写着 position_scale=1.2，也不会报错。
        """
        self.robot_origin = robot_origin.astype(float)

        if position_scale is not None:
            self.position_scale_xy = position_scale
        else:
            self.position_scale_xy = position_scale_xy

        self.depth_scale = depth_scale

        self.min_gripper_width = min_gripper_width
        self.max_gripper_width = max_gripper_width

        self.base_wrist_img = None
        self.base_palm_size = None

        self.pos_filter = LowPassFilter(alpha=filter_alpha)
        self.rot_filter = LowPassFilter(alpha=filter_alpha)

    def reset_origin(self):
        self.base_wrist_img = None
        self.base_palm_size = None
        self.pos_filter.reset()
        self.rot_filter.reset()

    @staticmethod
    def choose_landmarks(obs) -> np.ndarray:
        """
        兼容旧版 update() 的接口。
        优先返回 world landmarks，没有则返回 image landmarks。
        """
        if obs.landmarks_world is not None:
            return obs.landmarks_world.copy()
        return obs.landmarks_image.copy()

    @staticmethod
    def palm_frame(P: np.ndarray) -> np.ndarray:
        """
        用 21 个手部关键点构建手掌坐标系。

        0  : wrist
        5  : index MCP
        9  : middle MCP
        17 : pinky MCP
        """
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

        return R_hand

    @staticmethod
    def pinch_ratio(P: np.ndarray) -> float:
        """
        thumb tip 到 index tip 的距离 / 手掌尺度。
        数值越小，说明越接近捏合。
        """
        thumb_tip = P[4]
        index_tip = P[8]
        wrist = P[0]
        middle_mcp = P[9]

        pinch = np.linalg.norm(thumb_tip - index_tip)
        palm = np.linalg.norm(middle_mcp - wrist) + 1e-9

        return float(pinch / palm)

    def map_position_from_image(self, P_img: np.ndarray) -> np.ndarray:
        """
        用 image landmarks 做手部整体平移映射。

        P_img[:, 0]：图像 x，向右增大 → 映射到 robot y
        P_img[:, 1]：图像 y，向下增大 → 映射到 robot z（dy 取反）
        详细坐标系约定见 docs/retargeting.md
        """
        wrist = P_img[0]
        middle_mcp = P_img[9]

        palm_size = np.linalg.norm(middle_mcp[:2] - wrist[:2])

        if self.base_wrist_img is None:
            self.base_wrist_img = wrist.copy()
            self.base_palm_size = palm_size

        delta = wrist - self.base_wrist_img
        delta_palm = palm_size - self.base_palm_size

        dx = delta[0]
        dy = delta[1]

        robot_delta = np.array(
            [
                0.0,# self.depth_scale * delta_palm,        # 手靠近/远离摄像头 -> Panda x
                self.position_scale_xy * dx,         # 手左右移动 -> Panda y
                -0.8*self.position_scale_xy * dy,         # 手上下移动 -> Panda z
            ],
            dtype=float,
        )

        pos = self.robot_origin + robot_delta

        pos[0] = np.clip(pos[0], self.robot_origin[0] - 0.02, self.robot_origin[0] + 0.02)
        pos[1] = np.clip(pos[1], self.robot_origin[1] - 0.25, self.robot_origin[1] + 0.25)
        pos[2] = np.clip(pos[2], self.robot_origin[2] - 0.12, self.robot_origin[2] + 0.14)

        return pos

    def map_orientation(self, R_hand: np.ndarray) -> np.ndarray:
        """
        手掌坐标系到 Panda 末端坐标系的初始映射。
        这个矩阵后面可以根据实际观感继续调。
        """
        R_robot_from_hand = Rotation.from_euler(
            "xyz",
            [180, 0, 90],
            degrees=True,
        ).as_matrix()

        return R_robot_from_hand @ R_hand

    def map_gripper(self, pinch_ratio: float) -> float:
        """
        pinch ratio -> gripper width。
        """
        ratio_open = 1.2
        ratio_closed = 0.35

        s = (pinch_ratio - ratio_closed) / (ratio_open - ratio_closed)
        s = np.clip(s, 0.0, 1.0)

        return self.min_gripper_width + s * (
            self.max_gripper_width - self.min_gripper_width
        )

    def update(self, obs) -> PandaTarget:
        """
        主更新函数。
        注意：这个版本不再直接调用 choose_landmarks 做平移，
        因为平移要用 image landmarks，姿态才优先用 world landmarks。
        """
        P_img = obs.landmarks_image

        if P_img is None or P_img.shape != (21, 3):
            return PandaTarget(
                pos=self.robot_origin.copy(),
                rot=np.eye(3),
                gripper_width=self.max_gripper_width,
                pinch_ratio=1.0,
                valid=False,
            )

        # 1. 平移：用 image landmarks，更明显响应手在画面里的移动
        pos = self.map_position_from_image(P_img)

        # 2. 姿态和 pinch：优先用 world landmarks
        P_pose = obs.landmarks_world if obs.landmarks_world is not None else P_img

        if P_pose.shape != (21, 3):
            return PandaTarget(
                pos=pos,
                rot=np.eye(3),
                gripper_width=self.max_gripper_width,
                pinch_ratio=1.0,
                valid=False,
            )

        R_hand = self.palm_frame(P_pose)
        pinch = self.pinch_ratio(P_pose)

        R_des = self.map_orientation(R_hand)
        gripper = self.map_gripper(pinch)

        # 3. 低通滤波，降低抖动
        pos = self.pos_filter.update(pos)

        rotvec = Rotation.from_matrix(R_des).as_rotvec()
        rotvec = self.rot_filter.update(rotvec)
        R_des = Rotation.from_rotvec(rotvec).as_matrix()

        return PandaTarget(
            pos=pos,
            rot=R_des,
            gripper_width=gripper,
            pinch_ratio=pinch,
            valid=True,
        )

