# core/pose_ik.py

from dataclasses import dataclass
from typing import Optional

import mujoco
import numpy as np

from core.dynamics_control import (
    get_body_pose,
    get_body_jacobian,
    rotation_error_rotvec,
    damped_pinv,
)


@dataclass
class PoseIKResult:
    """Result of a pose IK solve."""

    q: np.ndarray
    success: bool
    num_iters: int
    pos_error_norm: float
    rot_error_norm: float


def solve_pose_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_pos: np.ndarray,
    target_rot: np.ndarray,
    body_name: str = "hand",
    dof: int = 7,
    q_init: Optional[np.ndarray] = None,
    max_iters: int = 100,
    damping: float = 1e-3,
    step_size: float = 0.5,
    pos_weight: float = 1.0,
    rot_weight: float = 1.0,
    tol: float = 1e-4,
) -> PoseIKResult:
    """
    Damped least-squares pose IK for a Panda-like arm.

    Iteratively solves::

        dq = J.T @ inv(J @ J.T + lambda^2 I) @ error

    where *error* is a 6D twist combining position and orientation error.
    Orientation error is computed via ``rotation_error_rotvec`` (rotvec
    representation).  Joint limits are clamped each iteration.

    Parameters
    ----------
    model, data : MuJoCo model/data.
    target_pos : shape (3,) desired end-effector position.
    target_rot : shape (3, 3) desired end-effector rotation matrix.
    body_name : MuJoCo body name for the end-effector (default ``"hand"``).
    dof : number of actuated joints (default 7).
    q_init : optional starting configuration; uses current ``data.qpos[:dof]``
        if not provided.
    max_iters : maximum Newton iterations.
    damping : DLS damping factor.
    step_size : step-size multiplier applied to each ``dq`` update.
    pos_weight : scaling factor for position error.
    rot_weight : scaling factor for orientation error.
    tol : convergence threshold on the norm of the 6D error.

    Returns
    -------
    PoseIKResult with final joint angles, success flag, iteration count,
    and final position/rotation error norms.
    """
    q = data.qpos[:dof].copy() if q_init is None else q_init.copy()

    jnt_range = model.jnt_range[:dof]
    has_limits = jnt_range.shape[0] == dof and not np.all(np.isinf(jnt_range))

    for iteration in range(max_iters):
        data.qpos[:dof] = q
        mujoco.mj_forward(model, data)

        pos_cur, R_cur = get_body_pose(model, data, body_name)
        J = get_body_jacobian(model, data, body_name, dof)

        pos_err = target_pos - pos_cur
        rot_err = rotation_error_rotvec(target_rot, R_cur)

        err_6d = np.concatenate([
            pos_weight * pos_err,
            rot_weight * rot_err,
        ])

        err_norm = np.linalg.norm(err_6d)
        if err_norm < tol:
            break

        dq = damped_pinv(J, damping=damping) @ err_6d
        q = q + step_size * dq

        if has_limits:
            q = np.clip(q, jnt_range[:, 0], jnt_range[:, 1])

    data.qpos[:dof] = q
    mujoco.mj_forward(model, data)

    pos_cur, R_cur = get_body_pose(model, data, body_name)
    pos_err = np.linalg.norm(target_pos - pos_cur)
    rot_err = np.linalg.norm(rotation_error_rotvec(target_rot, R_cur))
    success = pos_err < 0.05 and rot_err < 0.05

    return PoseIKResult(
        q=q,
        success=bool(success),
        num_iters=iteration + 1,
        pos_error_norm=float(pos_err),
        rot_error_norm=float(rot_err),
    )
