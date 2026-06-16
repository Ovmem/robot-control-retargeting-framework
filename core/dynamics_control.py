# core/dynamics_control.py

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
    """
    Orientation error represented as rotation vector.
    R_err maps current frame to desired frame.
    """
    R_err = R_des @ R_cur.T
    return Rotation.from_matrix(R_err).as_rotvec()


def get_body_pose(model: mujoco.MjModel, data: mujoco.MjData, body_name: str):
    """Return body position and rotation matrix."""
    body_id = model.body(body_name).id
    pos = data.xpos[body_id].copy()
    R = data.xmat[body_id].reshape(3, 3).copy()
    return pos, R


def get_body_jacobian(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
    dof: int = 7,
) -> np.ndarray:
    """
    Return 6 x dof spatial Jacobian of the given body.
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
    Check if the first *dof* actuators are affine-bias position actuators.

    In the standard Panda MJCF, arm joints use ``<general>`` actuators with
    ``dyntype="none"`` and ``biastype="affine"``.  These behave as position
    servos rather than pure torque sources.

    Returns ``True`` when the first actuator exhibits this pattern, which is
    sufficient for the homogeneous Panda arm actuator set.
    """
    if model.nu < 1:
        return False
    affine = int(mujoco.mjtBias.mjBIAS_AFFINE)
    return bool(model.actuator_biastype[0] == affine)


def neutralize_position_actuators(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    dof: int = 7,
):
    """
    Set ``data.ctrl[:dof] = data.qpos[:dof]`` so that affine-bias position
    actuators hold the current joint configuration instead of fighting
    externally applied torques (``qfrc_applied``).

    This is an *engineering compromise*: the standard Panda MJCF XML ships
    with position-level actuators, so performing torque-level control via
    ``qfrc_applied`` requires keeping the position servo at the current
    pose.  The cleanest fix would be to author a torque-actuated variant of
    the XML, but that is a separate task.

    Call this once at the start of each control step, *before*
    ``mujoco.mj_step`` and after writing ``qfrc_applied``.
    """
    data.ctrl[:dof] = data.qpos[:dof].copy()


def has_position_actuators_and_neutralize(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    dof: int = 7,
) -> bool:
    """
    Convenience: check for position actuators and neutralize them.

    Returns ``True`` if neutralization was applied.
    """
    if has_affine_position_actuators(model, dof):
        neutralize_position_actuators(model, data, dof)
        return True
    return False


@dataclass
class TorqueLimit:
    lower: np.ndarray
    upper: np.ndarray


class PandaTorqueController:
    """
    Torque controller for the first 7 DoF of Franka Panda.

    It supports:
    1. Joint-space PD.
    2. Joint-space PD + gravity/bias compensation.
    3. Task-space impedance-like control.
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
        if self.torque_limit is None:
            return tau

        return np.clip(tau, self.torque_limit.lower, self.torque_limit.upper)

    def bias_torque(self) -> np.ndarray:
        """
        Bias torque from MuJoCo.

        In a simple no-contact holding task, this can be used as the first
        version of gravity/Coriolis compensation.
        For pure gravity compensation, call this when qvel is zero.
        """
        mujoco.mj_forward(self.model, self.data)
        return self.data.qfrc_bias[: self.dof].copy()

    def gravity_comp_torque(self) -> np.ndarray:
        """
        Approximate gravity compensation by evaluating bias torque at qdot = 0.

        This modifies qvel temporarily, then restores it.
        """
        qd_backup = self.data.qvel.copy()

        self.data.qvel[: self.dof] = 0.0
        mujoco.mj_forward(self.model, self.data)
        tau_g = self.data.qfrc_bias[: self.dof].copy()

        self.data.qvel[:] = qd_backup
        mujoco.mj_forward(self.model, self.data)

        return tau_g

    def inverse_dynamics_torque(
        self,
        qacc_des: np.ndarray,
    ) -> np.ndarray:
        """
        Compute inverse dynamics torque for desired acceleration.

        This is useful for computed torque control:
            tau = M(q) qdd_des + C(q, qd) + g(q)
        """
        qacc_backup = self.data.qacc.copy()

        self.data.qacc[: self.dof] = qacc_des
        mujoco.mj_inverse(self.model, self.data)
        tau = self.data.qfrc_inverse[: self.dof].copy()

        self.data.qacc[:] = qacc_backup
        mujoco.mj_forward(self.model, self.data)

        return tau

    def joint_pd(
        self,
        q_des: np.ndarray,
        qd_des: Optional[np.ndarray] = None,
        kp: float | np.ndarray = 80.0,
        kd: float | np.ndarray = 8.0,
        gravity_comp: bool = True,
    ) -> np.ndarray:
        """
        Joint-space PD torque:
            tau = Kp(q_des - q) + Kd(qd_des - qd) + tau_g
        """
        if qd_des is None:
            qd_des = np.zeros(self.dof)

        q = self.q()
        qd = self.qd()

        tau = kp * (q_des - q) + kd * (qd_des - qd)

        if gravity_comp:
            tau += self.gravity_comp_torque()

        return self.clip_tau(tau)

    def computed_torque(
        self,
        q_des: np.ndarray,
        qd_des: Optional[np.ndarray] = None,
        qdd_des_ff: Optional[np.ndarray] = None,
        kp: float | np.ndarray = 80.0,
        kd: float | np.ndarray = 12.0,
    ) -> np.ndarray:
        """
        Computed torque control:
            qdd_cmd = qdd_ff + Kp(q_des-q) + Kd(qd_des-qd)
            tau = ID(q, qd, qdd_cmd)
        """
        if qd_des is None:
            qd_des = np.zeros(self.dof)

        if qdd_des_ff is None:
            qdd_des_ff = np.zeros(self.dof)

        q = self.q()
        qd = self.qd()

        qdd_cmd = qdd_des_ff + kp * (q_des - q) + kd * (qd_des - qd)

        return self.clip_tau(self.inverse_dynamics_torque(qdd_cmd))

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
        Task-space impedance-like torque:
            F = Kx * x_err + Dx * xdot_err
            tau = J.T F + N.T tau_null + tau_g
        """
        mujoco.mj_forward(self.model, self.data)

        pos, R = get_body_pose(self.model, self.data, self.body_name)
        J = get_body_jacobian(self.model, self.data, self.body_name, self.dof)

        q = self.q()
        qd = self.qd()

        pos_err = target_pos - pos
        rot_err = rotation_error_rotvec(target_rot, R)

        xdot = J @ qd
        vel_err = -xdot[:3]
        omega_err = -xdot[3:]

        wrench = np.zeros(6)
        wrench[:3] = kp_pos * pos_err + kd_pos * vel_err
        wrench[3:] = kp_rot * rot_err + kd_rot * omega_err

        tau_task = J.T @ wrench

        tau = tau_task

        if q_null_des is not None:
            # Simple torque-level null-space projection.
            J_T_pinv = damped_pinv(J.T, damping=damping)
            N_T = np.eye(self.dof) - J.T @ J_T_pinv

            tau_null = kp_null * (q_null_des - q) - kd_null * qd
            tau += N_T @ tau_null

        if gravity_comp:
            tau += self.gravity_comp_torque()

        return self.clip_tau(tau)

    def apply_torque(self, tau: np.ndarray, prefer_ctrl: bool = False):
        """
        Apply torque.

        If your XML has 7 motor actuators, prefer_ctrl=True writes to data.ctrl.
        Otherwise this writes directly to data.qfrc_applied for a clean algorithm demo.
        """
        tau = self.clip_tau(tau)

        if prefer_ctrl and self.model.nu >= self.dof:
            self.data.ctrl[: self.dof] = tau
        else:
            self.data.qfrc_applied[:] = 0.0
            self.data.qfrc_applied[: self.dof] = tau
