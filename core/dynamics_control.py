# core/dynamics_control.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def damped_pinv(J: np.ndarray, damping: float = 1e-3) -> np.ndarray:
    """Damped least-squares pseudo inverse."""
    m, n = J.shape
    if m <= n:
        return J.T @ np.linalg.inv(J @ J.T + damping**2 * np.eye(m))
    return np.linalg.inv(J.T @ J + damping**2 * np.eye(n)) @ J.T


def rotation_error_rotvec(R_des: np.ndarray, R_cur: np.ndarray) -> np.ndarray:
    """Return orientation error from current rotation to desired rotation."""
    R_err = R_des @ R_cur.T
    return Rotation.from_matrix(R_err).as_rotvec()


def get_body_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return body position and rotation matrix."""
    body_id = model.body(body_name).id
    pos = data.xpos[body_id].copy()
    rot = data.xmat[body_id].reshape(3, 3).copy()
    return pos, rot


def get_body_jacobian(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
    dof: int = 7,
) -> np.ndarray:
    """
    Return 6 x dof body Jacobian.

    First 3 rows: translational Jacobian.
    Last 3 rows: rotational Jacobian.
    """
    body_id = model.body(body_name).id
    Jp = np.zeros((3, model.nv))
    Jr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, Jp, Jr, body_id)
    return np.vstack([Jp[:, :dof], Jr[:, :dof]])


def has_affine_position_actuators(
    model: mujoco.MjModel,
    dof: int = 7,
) -> bool:
    """
    Check whether the first dof actuators are affine-bias position actuators.

    The original Panda MJCF commonly uses affine-bias general actuators for
    arm joint position servos. This demo's task-space torque control should
    use a torque-actuated XML instead.
    """
    if model.nu < dof:
        return False

    affine = int(mujoco.mjtBias.mjBIAS_AFFINE)
    return any(int(model.actuator_biastype[i]) == affine for i in range(dof))


def print_actuator_diagnostics(
    model: mujoco.MjModel,
    dof: int = 7,
) -> None:
    """Print actuator names and ctrl ranges for quick debugging."""
    print("\n=== Actuator diagnostics ===")
    print(f"nu = {model.nu}")
    for i in range(min(dof, model.nu)):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        name = name or f"actuator_{i}"
        dyntype = int(model.actuator_dyntype[i])
        biastype = int(model.actuator_biastype[i])
        ctrlrange = model.actuator_ctrlrange[i].copy()
        forcerange = model.actuator_forcerange[i].copy()
        print(
            f"[{i}] {name:>16s} | "
            f"dyntype={dyntype} biastype={biastype} | "
            f"ctrlrange={ctrlrange} | forcerange={forcerange}"
        )
    print("============================\n")


@dataclass
class TorqueLimit:
    lower: np.ndarray
    upper: np.ndarray

    @classmethod
    def panda_default(cls) -> "TorqueLimit":
        """
        Conservative Panda-like torque limits.

        The first four joints are allowed larger torques.
        The last three wrist joints are more limited.
        """
        tau = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], dtype=float)
        return cls(lower=-tau, upper=tau)

    @classmethod
    def uniform(cls, limit: float, dof: int = 7) -> "TorqueLimit":
        tau = np.full(dof, float(limit), dtype=float)
        return cls(lower=-tau, upper=tau)


class PandaTorqueController:
    """
    Torque-level controller for the first 7 DoF of Franka Panda.

    Important:
    - For torque-actuated XML, use apply_torque(..., prefer_ctrl=True).
    - For position-actuated XML, do not use this as the main controller.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        dof: int = 7,
        body_name: str = "hand",
        torque_limit: Optional[TorqueLimit] = None,
    ):
        self.model = model
        self.data = data
        self.dof = dof
        self.body_name = body_name
        self.torque_limit = torque_limit

    def q(self) -> np.ndarray:
        return self.data.qpos[: self.dof].copy()

    def qd(self) -> np.ndarray:
        return self.data.qvel[: self.dof].copy()

    def clip_tau(self, tau: np.ndarray) -> np.ndarray:
        tau = np.asarray(tau, dtype=float).copy()
        if self.torque_limit is None:
            return tau
        return np.clip(tau, self.torque_limit.lower, self.torque_limit.upper)

    def gravity_comp_torque(self) -> np.ndarray:
        """
        Approximate gravity compensation using MuJoCo bias force at zero qvel.

        This temporarily sets joint velocities to zero, evaluates qfrc_bias,
        then restores velocities.
        """
        qvel_backup = self.data.qvel.copy()

        self.data.qvel[: self.dof] = 0.0
        mujoco.mj_forward(self.model, self.data)
        tau_g = self.data.qfrc_bias[: self.dof].copy()

        self.data.qvel[:] = qvel_backup
        mujoco.mj_forward(self.model, self.data)
        return tau_g

    def task_space_pd(
        self,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        kp_pos: float = 250.0,
        kd_pos: float = 30.0,
        kp_rot: float = 40.0,
        kd_rot: float = 6.0,
        q_null_des: Optional[np.ndarray] = None,
        kp_null: float = 5.0,
        kd_null: float = 1.0,
        damping: float = 1e-3,
        gravity_comp: bool = True,
    ) -> np.ndarray:
        """
        Task-space PD / impedance-style torque.

        It computes:
        - end-effector position error
        - end-effector orientation error
        - spatial velocity through Jacobian
        - task wrench
        - joint torque through J.T @ wrench
        """
        mujoco.mj_forward(self.model, self.data)

        pos, rot = get_body_pose(self.model, self.data, self.body_name)
        J = get_body_jacobian(self.model, self.data, self.body_name, self.dof)

        q = self.q()
        qd = self.qd()

        pos_err = np.asarray(target_pos, dtype=float) - pos
        rot_err = rotation_error_rotvec(np.asarray(target_rot, dtype=float), rot)

        xdot = J @ qd
        vel_err = -xdot[:3]
        omega_err = -xdot[3:]

        wrench = np.zeros(6)
        wrench[:3] = kp_pos * pos_err + kd_pos * vel_err
        wrench[3:] = kp_rot * rot_err + kd_rot * omega_err

        tau = J.T @ wrench

        if q_null_des is not None:
            J_T_pinv = damped_pinv(J.T, damping=damping)
            N_T = np.eye(self.dof) - J.T @ J_T_pinv
            tau_null = kp_null * (q_null_des - q) - kd_null * qd
            tau += N_T @ tau_null

        if gravity_comp:
            tau += self.gravity_comp_torque()

        return self.clip_tau(tau)

    def joint_pd_gravity_comp(
        self,
        q_des: np.ndarray,
        kp: float | np.ndarray = 120.0,
        kd: float | np.ndarray = 12.0,
        gravity_comp: bool = True,
    ) -> np.ndarray:
        """Joint-space PD torque with optional MuJoCo gravity compensation."""
        q_des = np.asarray(q_des, dtype=float)
        kp_arr = np.asarray(kp, dtype=float)
        kd_arr = np.asarray(kd, dtype=float)

        tau = kp_arr * (q_des[: self.dof] - self.q()) - kd_arr * self.qd()
        if gravity_comp:
            tau = tau + self.gravity_comp_torque()

        return self.clip_tau(tau)

    def apply_torque(self, tau: np.ndarray, prefer_ctrl: bool = True) -> np.ndarray:
        """
        Apply torque to MuJoCo.

        prefer_ctrl=True:
            Use data.ctrl[:dof]. This is the correct path for torque motor XML.

        prefer_ctrl=False:
            Use data.qfrc_applied[:dof]. This is only for debugging without
            torque actuators and should not be used as the main Panda demo path.
        """
        tau = self.clip_tau(tau)

        # Always clear external forces to avoid stale qfrc_applied fighting ctrl.
        self.data.qfrc_applied[:] = 0.0

        if prefer_ctrl:
            if self.model.nu < self.dof:
                raise RuntimeError(
                    f"Model has only {self.model.nu} actuators, "
                    f"but torque ctrl requires at least {self.dof}."
                )
            self.data.ctrl[: self.dof] = tau
        else:
            self.data.ctrl[: self.dof] = 0.0
            self.data.qfrc_applied[: self.dof] = tau

        return tau


