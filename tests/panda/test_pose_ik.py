import pytest
import numpy as np

import mujoco

from core.pose_ik import solve_pose_ik, PoseIKResult


@pytest.mark.mujoco
def test_pose_ik_converges():
    """Verify that pose IK converges for a small position offset."""
    model = mujoco.MjModel.from_xml_path("models/panda/panda.xml")
    data = mujoco.MjData(model)

    q0 = np.array([0.0, -0.7, 0.0, -2.2, 0.0, 1.6, 0.8])
    data.qpos[:7] = q0
    mujoco.mj_forward(model, data)

    body_id = model.body("hand").id
    pos0 = data.xpos[body_id].copy()
    R0 = data.xmat[body_id].reshape(3, 3).copy()

    target_pos = pos0 + np.array([0.03, 0.0, 0.0])
    target_rot = R0

    result = solve_pose_ik(
        model,
        data,
        target_pos=target_pos,
        target_rot=target_rot,
        body_name="hand",
        dof=7,
        q_init=q0.copy(),
        max_iters=80,
        damping=5e-3,
        step_size=0.5,
        tol=1e-4,
    )

    assert isinstance(result, PoseIKResult), f"expected PoseIKResult, got {type(result)}"
    assert result.q.shape == (7,), f"q.shape={result.q.shape}"
    assert np.all(np.isfinite(result.q)), "q contains NaN or inf"
    assert result.pos_error_norm < 0.01, f"pos_error={result.pos_error_norm:.6f}"
    assert result.rot_error_norm < 0.01, f"rot_error={result.rot_error_norm:.6f}"


@pytest.mark.mujoco
def test_pose_ik_returns_valid_type():
    """Verify return type and field access for an identity target."""
    model = mujoco.MjModel.from_xml_path("models/panda/panda.xml")
    data = mujoco.MjData(model)

    q0 = np.array([0.0, -0.7, 0.0, -2.2, 0.0, 1.6, 0.8])
    data.qpos[:7] = q0
    mujoco.mj_forward(model, data)

    body_id = model.body("hand").id
    pos0 = data.xpos[body_id].copy()
    R0 = data.xmat[body_id].reshape(3, 3).copy()

    result = solve_pose_ik(
        model, data, target_pos=pos0, target_rot=R0,
        q_init=q0.copy(), max_iters=20, damping=1e-2,
    )

    assert isinstance(result, PoseIKResult)
    assert np.all(np.isfinite(result.q))
    assert result.pos_error_norm >= 0
    assert result.rot_error_norm >= 0
    assert 0 < result.num_iters <= 20
    assert isinstance(result.success, bool)


if __name__ == "__main__":
    test_pose_ik_converges()
    test_pose_ik_returns_valid_type()
    print("All pose IK tests passed.")
