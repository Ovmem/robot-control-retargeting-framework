# vision/hand_tracker.py

from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class HandObservation:
    frame_bgr: np.ndarray
    landmarks_image: np.ndarray      # shape: (21, 3), normalized image coordinates
    landmarks_world: Optional[np.ndarray]  # shape: (21, 3), metric world coordinates if available
    handedness: str
    score: float


class MediaPipeHandTracker:
    """
    MediaPipe Hands wrapper.

    Returns detection results without displaying windows.
    The caller (demo script) is responsible for showing the camera feed.
    """

    def __init__(
        self,
        camera_id: int = 0,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
        draw: bool = False,
        mirror: bool = True,
    ):
        self.draw = draw
        self.mirror = mirror

        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera {camera_id}. "
                "Please check --camera-id."
            )

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=1,
        )

    @staticmethod
    def _landmarks_to_np(landmark_list) -> np.ndarray:
        return np.array(
            [[lm.x, lm.y, lm.z] for lm in landmark_list.landmark],
            dtype=np.float64,
        )

    def read(self) -> Optional[HandObservation]:
        ret, frame = self.cap.read()
        if not ret:
            return None

        if self.mirror:
            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        if not result.multi_hand_landmarks:
            return None

        hand_landmarks = result.multi_hand_landmarks[0]
        landmarks_image = self._landmarks_to_np(hand_landmarks)

        landmarks_world = None
        if result.multi_hand_world_landmarks:
            landmarks_world = self._landmarks_to_np(result.multi_hand_world_landmarks[0])

        handedness = "Unknown"
        score = 0.0
        if result.multi_handedness:
            cls = result.multi_handedness[0].classification[0]
            handedness = cls.label
            score = cls.score

        if self.draw:
            self.mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
            )
            cv2.putText(
                frame,
                f"{handedness}: {score:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )

        return HandObservation(
            frame_bgr=frame,
            landmarks_image=landmarks_image,
            landmarks_world=landmarks_world,
            handedness=handedness,
            score=score,
        )

    def close(self):
        self.cap.release()
