# Hand Retargeting: Coordinate Frames, Mapping Pipeline & Constraints

## Overview

This document describes the engineering boundary of the current hand-to-Panda
retargeting prototype.  The system uses a webcam, MediaPipe Hands, and a
simplified mapping to control the Franka Panda end-effector target pose in
MuJoCo simulation.

**Scope:** hand → Panda end-effector target (position + orientation + gripper).
Not a full-body or VR motion retargeting system.

**Current status:** simulation prototype.  Not deployed on a real robot.

---

## 1.  Coordinate Frames

### A. Camera / Image Frame

- **Source:** a single RGB webcam.
- **MediaPipe landmarks** are reported as normalized image coordinates:
  `x` ∈ [0, 1] (rightward), `y` ∈ [0, 1] (downward), `z` ∈ [–1, 1] (relative
  depth estimate, not metric).
- The image frame is **mirrored** in the demo (`mirror=True`) so that wrist
  left/right motion feels natural to the user.
- **No calibrated camera intrinsics or extrinsics** are used.  The `z`
  component from MediaPipe is a per-landmark relative depth, not a metric
  distance.  Currently the axis mapping ignores the image `z` (depth mapping
  is commented out in `map_position_from_image`).

### B. Hand Local Frame

A right-handed orthonormal frame is constructed from world landmarks
(`landmarks_world`) when available:

```
wrist     = P[0]
index_mcp = P[5]
pinky_mcp = P[17]
mid_mcp   = P[9]

x_axis = normalize(index_mcp - pinky_mcp)
y_raw  = normalize(mid_mcp - wrist)
z_axis = normalize(cross(x_axis, y_raw))
y_axis = normalize(cross(z_axis, x_axis))
```

If world landmarks are not available the image landmarks are used as a
fallback, but this degrades the orientation estimate.

**Note:** This is a *simplified* hand frame — it does not represent a
calibrated human wrist coordinate frame.  The frame orientation may drift
when fingers curl or the hand rotates significantly.

### C. Robot Base Frame

- The Panda arm operates in MuJoCo's world frame.
- A **fixed origin** (`robot_origin`) is defined at the home end-effector pose.
- Hand motion in the image plane is mapped to Cartesian deltas **relative to
  this origin**:

  | Image axis | Robot axis | Scale |
  |---|---|---|
  | dx (right/left) | y (right/left) | `position_scale_xy` ≈ 2.2 |
  | dy (up/down) | z (up/down) | `–0.8 × position_scale_xy` |
  | depth | x (forward) | commented out |

- The resulting position is clamped to a safe workspace box centered on
  `robot_origin`.

### D. End-Effector Frame

- The hand frame is rotated by a fixed offset to align with the Panda
  end-effector: `R_robot_from_hand = R_euler([180°, 0°, 90°])`.
- The orientation is low-pass filtered to reduce jitter.
- **Current limitation:** orientation tracking is a prototype; when the hand
  rotates significantly the mapping may behave unexpectedly.

---

## 2.  Mapping Pipeline

```
Step 1: Capture
    Webcam → MediaPipe Hands → 21 landmarks (image + world)

Step 2: Normalize
    Landmarks relative to wrist (P[0]) or palm center.

Step 3: Convert to robot offset
    Image dx → robot y (scale × 2.2)
    Image dy → robot z (scale × –1.76)
    Palm frame → target rotation via R_robot_from_hand
    Pinch ratio → gripper width

Step 4: Clamp & filter
    Target position clamped inside Panda workspace.
    Low-pass filter applied to both position and rotation.

Step 5: Send to controller
    target_pos, target_rot → task-space PD / impedance-style torque controller.
```

---

## 3.  Safety / Constraints

| Constraint | Status | Details |
|---|---|---|
| Workspace limits | ✅ Implemented | `map_position_from_image` clamps x/y/z relative to `robot_origin`. |
| Low-pass filter | ✅ Implemented | `LowPassFilter` (exponential smoothing) on both pos and rot. Alpha ≈ 0.18 in the demo. |
| Joint limit check | ❌ Planned | The simulation respects Panda joint limits via MuJoCo physics; no explicit pre-check in the retargeter. |
| Target jump rejection | ❌ Planned | No velocity-based limit on `target_pos` deltas.  If hand tracking glitches the target can jump. |
| Velocity smoothing | ❌ Planned | No explicit velocity limiter; the low-pass filter provides mild smoothing. |
| Emergency stop | ⚠️ Viewer exit | The demo exits when the MuJoCo viewer closes or ESC is pressed in the MediaPipe window.  No dedicated e-stop. |
| Real robot safety | ❌ N/A | Simulation only.  No real-robot deployment or safety layer implemented. |

---

---

## 5.  Offline Mock Retargeting Demo

An offline demo script is available at ``scripts/generate_retargeting_demo.py``.

**What it does:**

- Generates a synthetic hand-landmark trajectory (wrist moving in a
  Lissajous-like pattern, pinch ratio cycling open-to-close).
- Feeds the trajectory through the same ``HandToPandaRetargeter`` pipeline
  used by the webcam demo.
- Saves CSV results and a Matplotlib figure to ``results/retargeting/``.

**What it validates:**

- Mapping-pipeline logic (position scaling, axis assignment, clamping).
- Gripper command mapping from pinch ratio.
- Low-pass filter behaviour over a multi-second trajectory.

**What it does NOT validate:**

- MediaPipe tracking quality or hand-detection robustness.
- Real-time latency.

Run it with:

```bash
python scripts/generate_retargeting_demo.py
```

No webcam, display, or MuJoCo viewer is required.

## 4.  Current Limitations (Summary)

- Not a full motion retargeting system (hand → end-effector only).
- No calibrated 3D camera or metric hand pose.
- Orientation mapping is prototype-grade; may behave unexpectedly in large
  rotations.
- Depth (fore/aft) mapping is currently disabled.
- No velocity limiting or target jump rejection.
- Gripper command is computed but not closed-loop controlled.
- Single-handed control only (left or right, not both).
- Simulation only — no real robot deployment.