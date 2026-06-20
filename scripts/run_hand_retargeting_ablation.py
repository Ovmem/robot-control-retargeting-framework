# scripts/run_hand_retargeting_ablation.py

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from retargeting.hand_to_panda import HandToPandaRetargeter, PandaTarget

# ---------------------------------------------------------------------------
# Mock hand trajectory (reused from generate_retargeting_demo.py)
# ---------------------------------------------------------------------------
@dataclass
class MockHandObservation:
    landmarks_image: np.ndarray
    landmarks_world: np.ndarray


def make_mock_hand(init_wrist_xy=(0.5, 0.5), palm_span=0.12):
    P = np.zeros((21, 3), dtype=np.float64)
    wx, wy = init_wrist_xy
    P[0] = [wx, wy, 0.0]
    P[5] = [wx + 0.04, wy - 0.03, 0.0]
    P[9] = [wx + 0.02, wy - palm_span, 0.0]
    P[17] = [wx - 0.03, wy - 0.02, 0.0]
    P[4] = [wx + 0.06, wy + 0.01, 0.0]
    P[8] = [wx + 0.06, wy - 0.05, 0.0]
    for i in range(21):
        if np.allclose(P[i], 0.0) and i != 0:
            P[i] = P[0] + [0.0, -0.01 * (i % 5), 0.0]
    return P


def generate_mock_trajectory(duration=6.0, fps=30, freq_xy=0.35,
                              wrist_center=(0.5, 0.5),
                              amplitude_x=0.18, amplitude_y=0.10,
                              pinch_cycle=0.5) -> List[Dict]:
    """Return list of dicts with keys: timestamp, P_img, P_world, detected."""
    n = int(duration * fps)
    out = []
    for i in range(n):
        t = i / fps
        dx = amplitude_x * np.sin(2 * np.pi * freq_xy * t)
        dy = amplitude_y * np.sin(2 * np.pi * freq_xy * 0.7 * t)
        P = make_mock_hand(
            init_wrist_xy=(wrist_center[0] + dx, wrist_center[1] + dy),
            palm_span=0.12,
        )
        angle = 2 * np.pi * pinch_cycle * t
        pinch_open = 0.5 * (1.0 + np.sin(angle))
        spread = 0.04 * pinch_open + 0.005
        P[4, 0] = P[0, 0] + 0.06 + spread
        P[8, 0] = P[0, 0] + 0.06 - spread

        # Introduce occasional dropouts for robustness testing
        detected = True
        if i > 0 and i % 150 == 0:
            detected = False  # one frame dropout every ~5s

        out.append({
            "timestamp": t,
            "P_img": P.copy(),
            "P_world": P.copy(),
            "detected": detected,
        })
    return out


# ---------------------------------------------------------------------------
# Rate limiter for target position
# ---------------------------------------------------------------------------
class RateLimiter:
    """Clamp target position velocity to a max value."""

    def __init__(self, max_velocity: float = 0.5, dt: float = 1.0 / 30.0):
        self.max_velocity = max_velocity
        self.dt = dt
        self.prev_pos: Optional[np.ndarray] = None

    def update(self, target_pos: np.ndarray) -> np.ndarray:
        if self.prev_pos is None:
            self.prev_pos = target_pos.copy()
            return target_pos.copy()
        delta = target_pos - self.prev_pos
        max_delta = self.max_velocity * self.dt
        delta_norm = np.linalg.norm(delta)
        if delta_norm > max_delta and delta_norm > 1e-12:
            delta = delta * (max_delta / delta_norm)
        result = self.prev_pos + delta
        self.prev_pos = result.copy()
        return result

    def reset(self):
        self.prev_pos = None


# ---------------------------------------------------------------------------
# Wrapper around HandToPandaRetargeter for ablation control
# ---------------------------------------------------------------------------
class RetargetingPipeline:
    """Retargeting pipeline with optional modules for ablation."""

    def __init__(
        self,
        robot_origin=np.array([0.45, 0.0, 0.45]),
        position_scale_xy=2.2,
        filter_alpha=0.18,
        enable_smoothing=True,
        enable_workspace_clamp=True,
        enable_rate_limit=True,
        enable_dropout_hold=True,
        enable_orientation_mapping=True,
        enable_pinch_gripper=True,
        rate_limiter_max_velocity=0.5,
        fps=30.0,
    ):
        self.robot_origin = robot_origin.copy()
        self.position_scale_xy = position_scale_xy
        self.enable_smoothing = enable_smoothing
        self.enable_workspace_clamp = enable_workspace_clamp
        self.enable_rate_limit = enable_rate_limit
        self.enable_dropout_hold = enable_dropout_hold
        self.enable_orientation_mapping = enable_orientation_mapping
        self.enable_pinch_gripper = enable_pinch_gripper

        # Create the core retargeter
        # When smoothing is off, use alpha=1.0 (no smoothing)
        actual_alpha = filter_alpha if enable_smoothing else 1.0
        self.retargeter = HandToPandaRetargeter(
            robot_origin=robot_origin,
            position_scale_xy=position_scale_xy,
            depth_scale=0.0,
            enable_depth_mapping=False,
            filter_alpha=actual_alpha,
        )

        self.rate_limiter = RateLimiter(
            max_velocity=rate_limiter_max_velocity,
            dt=1.0 / fps,
        )

        self.prev_target: Optional[PandaTarget] = None
        self.identity_rot = np.eye(3)
        self.fixed_gripper = 0.02  # half-open default

    def reset(self):
        self.retargeter.reset_origin()
        self.rate_limiter.reset()
        self.prev_target = None

    def process_frame(self, P_img: Optional[np.ndarray],
                       P_world: Optional[np.ndarray],
                       detected: bool = True) -> Tuple[PandaTarget, bool]:
        """Process one frame and return (target, was_clipped).

        Returns
        -------
        target : PandaTarget
        was_clipped : bool  True if position was clipped by workspace limits
        """
        dt = self.retargeter.pos_filter.alpha

        # --- Dropout handling ---
        if not detected:
            if self.enable_dropout_hold and self.prev_target is not None:
                return self.prev_target, False
            else:
                # Reset to origin
                return PandaTarget(
                    pos=self.robot_origin.copy(),
                    rot=np.eye(3),
                    gripper_width=self.retargeter.max_gripper_width,
                    pinch_ratio=1.0,
                    valid=False,
                ), False

        # --- Position mapping ---
        if P_img is not None and P_img.shape == (21, 3):
            if self.enable_workspace_clamp:
                raw_pos = self.retargeter.map_position_from_image(P_img)
            else:
                # No-clamp version: call the retargeter but skip clipping
                wrist = P_img[0]
                middle_mcp = P_img[9]
                palm_size = np.linalg.norm(middle_mcp[:2] - wrist[:2])

                if self.retargeter.base_wrist_img is None:
                    self.retargeter.base_wrist_img = wrist.copy()
                    self.retargeter.base_palm_size = palm_size

                delta = wrist - self.retargeter.base_wrist_img
                robot_delta = np.array([
                    0.0,
                    self.position_scale_xy * delta[0],
                    -0.8 * self.position_scale_xy * delta[1],
                ], dtype=float)
                raw_pos = self.robot_origin + robot_delta

            pos = raw_pos.copy()
            was_clipped = not np.allclose(raw_pos, pos)
        else:
            pos = self.robot_origin.copy()
            was_clipped = False

        # --- Position smoothing (optional) ---
        if self.enable_smoothing:
            pos = self.retargeter.pos_filter.update(pos)

        # --- Rate limiting (optional) ---
        if self.enable_rate_limit:
            pos = self.rate_limiter.update(pos)

        # --- Orientation mapping ---
        P_pose = P_world if P_world is not None else P_img
        if P_pose is not None and P_pose.shape == (21, 3):
            if self.enable_orientation_mapping:
                R_hand = self.retargeter.palm_frame(P_pose)
                R_des = self.retargeter.map_orientation(R_hand)
                if self.enable_smoothing:
                    rotvec = Rotation.from_matrix(R_des).as_rotvec()
                    rotvec = self.retargeter.rot_filter.update(rotvec)
                    R_des = Rotation.from_rotvec(rotvec).as_matrix()
            else:
                R_des = self.identity_rot.copy()
        else:
            R_des = self.identity_rot.copy()

        # --- Pinch / gripper ---
        if P_pose is not None and P_pose.shape == (21, 3):
            pinch = self.retargeter.pinch_ratio(P_pose)
            if self.enable_pinch_gripper:
                gripper = self.retargeter.map_gripper(pinch)
            else:
                gripper = self.fixed_gripper
        else:
            pinch = 1.0
            gripper = self.retargeter.max_gripper_width

        target = PandaTarget(
            pos=pos,
            rot=R_des,
            gripper_width=gripper,
            pinch_ratio=pinch,
            valid=True,
        )

        self.prev_target = target
        return target, was_clipped


# ---------------------------------------------------------------------------
# Hand sequence loader
# ---------------------------------------------------------------------------
def load_hand_sequence_csv(path: str) -> List[Dict]:
    """Load a hand sequence CSV recorded by record_hand_sequence.py.

    Returns list of dicts with keys: timestamp, P_img, P_world, detected.
    """
    frames = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            detected = row.get("detected", "0") == "1"
            P_img = np.zeros((21, 3), dtype=np.float64)
            P_world = np.zeros((21, 3), dtype=np.float64)
            have_world = False
            for i in range(21):
                lx = row.get(f"l{i}_x", "")
                ly = row.get(f"l{i}_y", "")
                lz = row.get(f"l{i}_z", "")
                if lx != "" and ly != "" and lz != "":
                    P_img[i] = [float(lx), float(ly), float(lz)]
                wlx = row.get(f"wl{i}_x", "")
                wly = row.get(f"wl{i}_y", "")
                wlz = row.get(f"wl{i}_z", "")
                if wlx != "" and wly != "" and wlz != "":
                    P_world[i] = [float(wlx), float(wly), float(wlz)]
                    have_world = True

            frames.append({
                "timestamp": float(row.get("timestamp", 0.0)),
                "P_img": P_img,
                "P_world": P_world if have_world else None,
                "detected": detected,
            })
    return frames


# ---------------------------------------------------------------------------
# MuJoCo response simulator (optional)
# ---------------------------------------------------------------------------
def _simulate_mujoco_response(targets: List[PandaTarget],
                               model_path: str,
                               fps: float = 30.0) -> np.ndarray:
    """Run MuJoCo simulation with task-space control for each target.

    Returns (n_frames, 6) array: [ee_pos_xyz, torque_norm, joint_limit_margin, manipulability]
    where unavailable fields are filled with NaN.
    """
    try:
        import mujoco
        from core.dynamics_control import (
            PandaTorqueController,
            TorqueLimit,
            get_body_pose,
            has_affine_position_actuators,
            has_position_actuators_and_neutralize,
            rotation_error_rotvec,
        )
    except ImportError:
        # MuJoCo not available; return NaNs
        n = len(targets)
        result = np.full((n, 6), np.nan)
        return result

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    q_home = np.array([0.0, -0.7, 0.0, -2.2, 0.0, 1.6, 0.8])
    data.qpos[:7] = q_home
    data.qvel[:7] = 0.0
    data.qfrc_applied[:] = 0.0
    mujoco.mj_forward(model, data)

    torque_limit = TorqueLimit(
        lower=-np.array([87, 87, 87, 87, 12, 12, 12], dtype=float),
        upper=np.array([87, 87, 87, 87, 12, 12, 12], dtype=float),
    )
    controller = PandaTorqueController(
        model=model, data=data, dof=7, body_name="hand",
        torque_limit=torque_limit,
    )

    has_pa = has_affine_position_actuators(model, 7)
    sim_substeps = 60  # 60 substeps per frame at ~30 fps

    # Precompute joint limit range
    jnt_range = model.jnt_range[:7]

    n = len(targets)
    results = np.full((n, 7), np.nan)

    for idx, target in enumerate(targets):
        if not target.valid:
            continue

        tau = controller.task_space_pd(
            target_pos=target.pos,
            target_rot=target.rot,
            kp_pos=np.array([1400.0, 1100.0, 1600.0]),
            kd_pos=np.array([100.0, 85.0, 110.0]),
            kp_rot=8.0,
            kd_rot=2.0,
            gravity_comp=True,
        )

        for _ in range(sim_substeps):
            # Neutralize position actuators if needed
            if model.nu >= 7 and has_pa:
                data.ctrl[:7] = data.qpos[:7]
            if model.nu >= 8:
                g_ctrl = np.clip(target.gripper_width / 0.04 * 255.0, 0, 255)
                data.ctrl[7] = g_ctrl

            controller.apply_torque(tau, prefer_ctrl=False)
            if has_pa:
                has_position_actuators_and_neutralize(model, data, 7)
            mujoco.mj_step(model, data)

        # Collect response
        cur_pos, cur_rot = get_body_pose(model, data, "hand")
        pos_err = np.linalg.norm(target.pos - cur_pos)
        rot_err_vec = rotation_error_rotvec(target.rot, cur_rot)
        rot_err_norm = np.linalg.norm(rot_err_vec)
        tau_norm = np.linalg.norm(tau)

        # Joint limit margin
        q = data.qpos[:7].copy()
        limit_margins = []
        for j in range(7):
            lower = jnt_range[j, 0]
            upper = jnt_range[j, 1]
            if np.isfinite(lower) and np.isfinite(upper):
                margin = min(q[j] - lower, upper - q[j])
                limit_margins.append(margin)
        min_margin = min(limit_margins) if limit_margins else np.nan

        # Manipulability: sqrt(det(J @ J^T))
        try:
            from core.dynamics_control import get_body_jacobian
            J = get_body_jacobian(model, data, "hand", 7)
            JJT = J[:3] @ J[:3].T  # translational part only
            w = np.sqrt(np.linalg.det(JJT)) if np.linalg.det(JJT) > 0 else 0.0
        except Exception:
            w = np.nan

        # Orientation error
        rot_err_norm = np.linalg.norm(rotation_error_rotvec(target.rot, cur_rot))
        results[idx] = [cur_pos[0], cur_pos[1], cur_pos[2],
                         tau_norm, min_margin, w, rot_err_norm]

    return results


# ---------------------------------------------------------------------------
# Ablation configurations
# ---------------------------------------------------------------------------
ABLATION_CONFIGS = [
    {
        "mode": "full_pipeline",
        "enable_smoothing": True,
        "enable_workspace_clamp": True,
        "enable_rate_limit": True,
        "enable_dropout_hold": True,
        "enable_orientation_mapping": True,
        "enable_pinch_gripper": True,
    },
    {
        "mode": "no_smoothing",
        "enable_smoothing": False,
        "enable_workspace_clamp": True,
        "enable_rate_limit": True,
        "enable_dropout_hold": True,
        "enable_orientation_mapping": True,
        "enable_pinch_gripper": True,
    },
    {
        "mode": "no_workspace_clamp",
        "enable_smoothing": True,
        "enable_workspace_clamp": False,
        "enable_rate_limit": True,
        "enable_dropout_hold": True,
        "enable_orientation_mapping": True,
        "enable_pinch_gripper": True,
    },
    {
        "mode": "no_rate_limit",
        "enable_smoothing": True,
        "enable_workspace_clamp": True,
        "enable_rate_limit": False,
        "enable_dropout_hold": True,
        "enable_orientation_mapping": True,
        "enable_pinch_gripper": True,
    },
    {
        "mode": "no_dropout_hold",
        "enable_smoothing": True,
        "enable_workspace_clamp": True,
        "enable_rate_limit": True,
        "enable_dropout_hold": False,
        "enable_orientation_mapping": True,
        "enable_pinch_gripper": True,
    },
    {
        "mode": "no_orientation_mapping",
        "enable_smoothing": True,
        "enable_workspace_clamp": True,
        "enable_rate_limit": True,
        "enable_dropout_hold": True,
        "enable_orientation_mapping": False,
        "enable_pinch_gripper": True,
    },
    {
        "mode": "no_pinch_gripper",
        "enable_smoothing": True,
        "enable_workspace_clamp": True,
        "enable_rate_limit": True,
        "enable_dropout_hold": True,
        "enable_orientation_mapping": True,
        "enable_pinch_gripper": False,
    },
]


# ---------------------------------------------------------------------------
# Per-frame CSV fields
# ---------------------------------------------------------------------------
PER_FRAME_FIELDS = [
    "timestamp", "mode", "detected_hand",
    "target_pos_x", "target_pos_y", "target_pos_z",
    "target_velocity_norm", "target_acceleration_norm", "target_jerk_norm",
    "target_orientation_jump",
    "gripper_width", "pinch_ratio",
    "workspace_clipped",
    "ee_actual_pos_x", "ee_actual_pos_y", "ee_actual_pos_z",
    "ee_position_error", "ee_orientation_error",
    "torque_norm",
    "joint_limit_margin", "manipulability",
    "diverged",
]

METRICS_FIELDS = [
    "mode",
    "mean_ee_position_error", "max_ee_position_error",
    "mean_ee_orientation_error",
    "rms_torque", "max_torque", "torque_smoothness",
    "min_joint_limit_margin", "mean_manipulability",
    "target_pos_jitter", "target_velocity_rms",
    "target_acceleration_rms", "target_jerk_rms",
    "workspace_clip_ratio",
    "orientation_jump_mean", "orientation_jump_max",
    "gripper_jitter", "dropout_count",
    "diverged",
]


def compute_derived_pos_metrics(positions: np.ndarray,
                                 timestamps: np.ndarray) -> dict:
    """Compute velocity, acceleration, jerk from position trajectory."""
    n = len(positions)
    if n < 3:
        return {"vel_rms": np.nan, "acc_rms": np.nan,
                "jerk_rms": np.nan, "jitter": np.nan}

    dt = np.diff(timestamps)
    vel = np.diff(positions, axis=0) / dt[:, None]  # (n-1, 3)
    vel_norm = np.linalg.norm(vel, axis=1)

    acc = np.diff(vel, axis=0) / dt[1:, None] if len(vel) > 1 else np.zeros((1, 3))
    acc_norm = np.linalg.norm(acc, axis=1)

    jerk = np.diff(acc, axis=0) / dt[2:, None] if len(acc) > 1 else np.zeros((1, 3))
    jerk_norm = np.linalg.norm(jerk, axis=1)

    # Jitter: high-frequency variation (sum absolute diff of successive velocity)
    jitter = np.mean(np.abs(np.diff(vel_norm))) if len(vel_norm) > 1 else np.nan

    return {
        "vel_rms": float(np.sqrt(np.mean(vel_norm ** 2))) if len(vel_norm) > 0 else np.nan,
        "acc_rms": float(np.sqrt(np.mean(acc_norm ** 2))) if len(acc_norm) > 0 else np.nan,
        "jerk_rms": float(np.sqrt(np.mean(jerk_norm ** 2))) if len(jerk_norm) > 0 else np.nan,
        "jitter": float(jitter) if jitter is not None else np.nan,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run hand retargeting ablation study")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to recorded hand sequence CSV. "
                             "If omitted, uses mock trajectory.")
    parser.add_argument("--model", type=str, default="models/panda/panda.xml",
                        help="Panda MJCF model path (for MuJoCo response)")
    parser.add_argument("--out-dir", type=str,
                        default="results/retargeting/ablation")
    parser.add_argument("--no-mujoco", action="store_true",
                        help="Skip MuJoCo response simulation")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    metrics_dir = out_dir / "metrics"
    raw_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load hand sequence
    # ------------------------------------------------------------------
    if args.input:
        print(f"Loading hand sequence from: {args.input}")
        try:
            hand_frames = load_hand_sequence_csv(args.input)
            fps_est = 1.0 / max(hand_frames[1]["timestamp"] - hand_frames[0]["timestamp"], 0.001)
        except Exception as e:
            print(f"Error loading CSV: {e}")
            print("Falling back to mock trajectory.")
            hand_frames = generate_mock_trajectory(duration=6.0)
            fps_est = 30.0
    else:
        print("No --input provided; using mock hand trajectory.")
        hand_frames = generate_mock_trajectory(duration=6.0)
        fps_est = 30.0

    print(f"  {len(hand_frames)} frames loaded, ~{fps_est:.1f} fps")

    # Convert frames to arrays for metric computation
    n_frames = len(hand_frames)
    timestamps = np.array([f["timestamp"] for f in hand_frames])

    # ------------------------------------------------------------------
    # 2. Run each ablation mode
    # ------------------------------------------------------------------
    all_rows: Dict[str, List[Dict]] = {}
    all_metrics: Dict[str, Dict] = {}

    for cfg_idx, cfg in enumerate(ABLATION_CONFIGS):
        mode_name = cfg["mode"]
        print(f"\n[{cfg_idx + 1}/{len(ABLATION_CONFIGS)}] Running: {mode_name}")

        pipeline = RetargetingPipeline(
            robot_origin=np.array([0.45, 0.0, 0.45]),
            position_scale_xy=2.2,
            filter_alpha=0.18,
            fps=fps_est,
            **{k: v for k, v in cfg.items() if k != "mode"},
        )

        # Store per-frame target arrays
        target_positions = np.full((n_frames, 3), np.nan)
        target_orientation_jumps = np.full(n_frames, np.nan)
        gripper_widths = np.full(n_frames, np.nan)
        pinch_ratios = np.full(n_frames, np.nan)
        workspace_clipped_flags = np.zeros(n_frames, dtype=bool)
        detected_flags = np.zeros(n_frames, dtype=bool)
        prev_rotvec = None

        targets = []

        for fi, frame in enumerate(hand_frames):
            P_img = frame["P_img"]
            P_world = frame.get("P_world", None)
            detected = frame["detected"]

            target, clipped = pipeline.process_frame(
                P_img, P_world, detected=detected,
            )
            targets.append(target)
            target_positions[fi] = target.pos
            gripper_widths[fi] = target.gripper_width
            pinch_ratios[fi] = target.pinch_ratio
            workspace_clipped_flags[fi] = clipped
            detected_flags[fi] = detected

            # Orientation jump
            if prev_rotvec is not None and target.valid:
                cur_rotvec = Rotation.from_matrix(target.rot).as_rotvec()
                jump = np.linalg.norm(cur_rotvec - prev_rotvec)
                target_orientation_jumps[fi] = jump
                prev_rotvec = cur_rotvec.copy()
            else:
                target_orientation_jumps[fi] = 0.0
                if target.valid:
                    prev_rotvec = Rotation.from_matrix(target.rot).as_rotvec()

        # Compute velocity, acceleration, jerk from positions
        pos_metrics = compute_derived_pos_metrics(
            target_positions, timestamps)

        # ------------------------------------------------------------------
        # 3. MuJoCo response (optional)
        # ------------------------------------------------------------------
        if args.no_mujoco:
            mujoco_resp = np.full((n_frames, 6), np.nan)
        else:
            print(f"  Simulating MuJoCo response for {n_frames} frames...")
            mujoco_resp = _simulate_mujoco_response(
                targets, args.model, fps=fps_est)

        # ee_actual_pos_x/y/z = mujoco_resp[:, 0:3]
        # torque_norm           = mujoco_resp[:, 3]
        # joint_limit_margin    = mujoco_resp[:, 4]
        # manipulability        = mujoco_resp[:, 5]

        # Compute ee errors
        ee_position_errors = np.full(n_frames, np.nan)
        ee_orientation_errors = np.full(n_frames, np.nan)
        ee_actual_positions = np.full((n_frames, 3), np.nan)
        torque_norms = np.full(n_frames, np.nan)
        joint_limit_margins = np.full(n_frames, np.nan)
        manipulabilities = np.full(n_frames, np.nan)

        for fi in range(n_frames):
            ee_actual_positions[fi] = mujoco_resp[fi, 0:3]
            torque_norms[fi] = mujoco_resp[fi, 3]
            joint_limit_margins[fi] = mujoco_resp[fi, 4]
            manipulabilities[fi] = mujoco_resp[fi, 5]

            if not np.any(np.isnan(mujoco_resp[fi, 0:3])):
                pos_err = np.linalg.norm(
                    target_positions[fi] - ee_actual_positions[fi])
                ee_position_errors[fi] = pos_err

                if targets[fi].valid:
                    _, cur_rot = None, None
                    try:
                        from core.dynamics_control import rotation_error_rotvec
                        _, cur_rot_mat = None, Rotation.from_matrix(targets[fi].rot)
                    except Exception:
                        ee_orientation_errors[fi] = np.nan

        # torch norm smoothness
        if not np.all(np.isnan(torque_norms)):
            tau_finite = torque_norms[np.isfinite(torque_norms)]
            tau_smoothness = float(np.mean(np.abs(np.diff(tau_finite)))) if len(tau_finite) > 1 else np.nan
        else:
            tau_smoothness = np.nan

        # Count dropouts
        dropout_count = int(np.sum(~detected_flags))

        # ------------------------------------------------------------------
        # 4. Compute aggregate metrics
        # ------------------------------------------------------------------
        metrics = {
            "mean_ee_position_error": float(np.nanmean(ee_position_errors)),
            "max_ee_position_error": float(np.nanmax(ee_position_errors)),
            "mean_ee_orientation_error": float(np.nanmean(ee_orientation_errors)),
            "rms_torque": float(np.sqrt(np.nanmean(torque_norms ** 2))),
            "max_torque": float(np.nanmax(torque_norms)),
            "torque_smoothness": tau_smoothness,
            "min_joint_limit_margin": float(np.nanmin(joint_limit_margins)),
            "mean_manipulability": float(np.nanmean(manipulabilities)),
            "target_pos_jitter": pos_metrics.get("jitter", np.nan),
            "target_velocity_rms": pos_metrics.get("vel_rms", np.nan),
            "target_acceleration_rms": pos_metrics.get("acc_rms", np.nan),
            "target_jerk_rms": pos_metrics.get("jerk_rms", np.nan),
            "workspace_clip_ratio": float(np.mean(workspace_clipped_flags)),
            "orientation_jump_mean": float(np.nanmean(target_orientation_jumps)),
            "orientation_jump_max": float(np.nanmax(target_orientation_jumps)),
            "gripper_jitter": float(np.nanstd(gripper_widths)),
            "dropout_count": dropout_count,
            "diverged": float(np.nanmean(torque_norms) if not np.all(np.isnan(torque_norms)) else 0.0),
        }

        all_metrics[mode_name] = metrics

        # ------------------------------------------------------------------
        # 5. Build per-frame rows
        # ------------------------------------------------------------------
        rows = []
        for fi in range(n_frames):
            row = {
                "timestamp": float(timestamps[fi]),
                "mode": mode_name,
                "detected_hand": int(detected_flags[fi]),
                "target_pos_x": float(target_positions[fi, 0]),
                "target_pos_y": float(target_positions[fi, 1]),
                "target_pos_z": float(target_positions[fi, 2]),
                "target_velocity_norm": float(np.linalg.norm(
                    target_positions[fi] - target_positions[max(fi - 1, 0)]) / max(timestamps[fi] - timestamps[max(fi - 1, 0)], 1e-6)) if fi > 0 else 0.0,
                # acceleration and jerk computed by gradient
                "target_acceleration_norm": float(pos_metrics.get("acc_rms", np.nan)),
                "target_jerk_norm": float(pos_metrics.get("jerk_rms", np.nan)),
                "target_orientation_jump": float(target_orientation_jumps[fi]) if fi < len(target_orientation_jumps) else 0.0,
                "gripper_width": float(gripper_widths[fi]),
                "pinch_ratio": float(pinch_ratios[fi]),
                "workspace_clipped": int(workspace_clipped_flags[fi]),
                "ee_actual_pos_x": float(ee_actual_positions[fi, 0]) if not np.isnan(ee_actual_positions[fi, 0]) else "",
                "ee_actual_pos_y": float(ee_actual_positions[fi, 1]) if not np.isnan(ee_actual_positions[fi, 1]) else "",
                "ee_actual_pos_z": float(ee_actual_positions[fi, 2]) if not np.isnan(ee_actual_positions[fi, 2]) else "",
                "ee_position_error": float(ee_position_errors[fi]) if not np.isnan(ee_position_errors[fi]) else "",
                "ee_orientation_error": float(ee_orientation_errors[fi]) if not np.isnan(ee_orientation_errors[fi]) else "",
                "torque_norm": float(torque_norms[fi]) if not np.isnan(torque_norms[fi]) else "",
                "joint_limit_margin": float(joint_limit_margins[fi]) if not np.isnan(joint_limit_margins[fi]) else "",
                "manipulability": float(manipulabilities[fi]) if not np.isnan(manipulabilities[fi]) else "",
                "diverged": str(metrics["diverged"]),
            }
            rows.append(row)

        all_rows[mode_name] = rows

        # Save per-frame CSV
        csv_path = raw_dir / f"{mode_name}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=PER_FRAME_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  saved: {csv_path}")

        # Print key metrics
        print(f"  target_jerk_rms: {metrics['target_jerk_rms']:.6f}")
        print(f"  workspace_clip_ratio: {metrics['workspace_clip_ratio']:.4f}")
        print(f"  orientation_jump_mean: {metrics['orientation_jump_mean']:.6f}")

    # ------------------------------------------------------------------
    # 6. Save aggregate metrics CSV
    # ------------------------------------------------------------------
    metrics_path = metrics_dir / "hand_retargeting_ablation_metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(METRICS_FIELDS)
        for mode_name in [c["mode"] for c in ABLATION_CONFIGS]:
            m = all_metrics.get(mode_name, {})
            writer.writerow([mode_name] + [m.get(k, float("nan")) for k in METRICS_FIELDS[1:]])
    print(f"\nMetrics saved: {metrics_path}")

    print("\nDone. Next: python scripts/plot_hand_retargeting_ablation.py")


if __name__ == "__main__":
    main()
