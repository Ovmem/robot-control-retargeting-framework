# tests/panda/test_retargeting_mapping.py

import pytest
import numpy as np

from retargeting.hand_to_panda import HandToPandaRetargeter, LowPassFilter


@pytest.fixture
def retargeter():
    return HandToPandaRetargeter(
        robot_origin=np.array([0.45, 0.0, 0.45]),
        position_scale_xy=2.2,
        depth_scale=0.0,
        filter_alpha=0.18,
    )


def _make_landmarks(wrist_xy=(0.5, 0.5), palm_span=0.12) -> np.ndarray:
    """Helper: build a (21, 3) landmark array usable by the retargeter."""
    P = np.zeros((21, 3), dtype=np.float64)
    wx, wy = wrist_xy
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


class TestMapPosition:
    """map_position_from_image – shape, finiteness, workspace clamp."""

    def test_output_shape_and_finite(self, retargeter):
        P = _make_landmarks()
        pos = retargeter.map_position_from_image(P)
        assert pos.shape == (3,)
        assert np.all(np.isfinite(pos))

    def test_workspace_bounds(self, retargeter):
        # First call sets baseline; second call moves wrist far
        P0 = _make_landmarks(wrist_xy=(0.5, 0.5))
        _ = retargeter.map_position_from_image(P0)       # baseline

        P_far = _make_landmarks(wrist_xy=(0.9, 0.9))
        pos = retargeter.map_position_from_image(P_far)

        origin = retargeter.robot_origin
        # x is clamped tightly (±0.02)
        assert origin[0] - 0.021 <= pos[0] <= origin[0] + 0.021
        # y ±0.25
        assert origin[1] - 0.26 <= pos[1] <= origin[1] + 0.26
        # z clamped accordingly
        assert origin[2] - 0.13 <= pos[2] <= origin[2] + 0.15


class TestLowPassFilter:
    def test_first_call_returns_input(self):
        flt = LowPassFilter(alpha=0.5)
        x = np.array([1.0, 2.0, 3.0])
        out = flt.update(x)
        assert np.allclose(out, x)
        assert out.shape == (3,)

    def test_convergence(self):
        flt = LowPassFilter(alpha=0.2)
        x = np.array([10.0, 0.0])
        out = flt.update(x)           # first call = x
        assert np.allclose(out, x)
        for _ in range(50):
            out = flt.update(x)
        # after many identical inputs the output should converge to x
        assert np.allclose(out, x, atol=1e-6)


class TestRetargeterUpdate:
    """End-to-end: update() with a mock observation."""

    def test_returns_valid_target(self, retargeter):
        P = _make_landmarks()
        obs = type("MockObs", (), {"landmarks_image": P, "landmarks_world": P})()
        target = retargeter.update(obs)
        assert target.pos.shape == (3,)
        assert target.rot.shape == (3, 3)
        assert np.all(np.isfinite(target.pos))
        assert np.all(np.isfinite(target.rot))
        assert 0.0 <= target.gripper_width <= 0.04
        assert isinstance(target.valid, bool)

    def test_position_changes_with_wrist_motion(self, retargeter):
        P0 = _make_landmarks(wrist_xy=(0.5, 0.5))
        obs0 = type("MockObs", (), {"landmarks_image": P0, "landmarks_world": P0})()
        _ = retargeter.update(obs0)   # baseline

        P1 = _make_landmarks(wrist_xy=(0.55, 0.45))
        obs1 = type("MockObs", (), {"landmarks_image": P1, "landmarks_world": P1})()
        target1 = retargeter.update(obs1)

        P2 = _make_landmarks(wrist_xy=(0.50, 0.50))
        obs2 = type("MockObs", (), {"landmarks_image": P2, "landmarks_world": P2})()
        target2 = retargeter.update(obs2)

        # target1 should differ from target2 due to wrist movement
        assert not np.allclose(target1.pos, target2.pos, atol=1e-4)



class TestGripperPinch:
    @pytest.fixture
    def retargeter(self):
        return HandToPandaRetargeter(robot_origin=np.array([0.45,0.0,0.45]),position_scale_xy=2.2,depth_scale=0.0,filter_alpha=0.18)

    def _lm(self, wx, spread):
        P = _make_landmarks(wrist_xy=(wx,0.5), palm_span=0.12)
        P[4,0] = wx + 0.06 + spread/2.0
        P[8,0] = wx + 0.06 - spread/2.0
        return P

    def test_open_vs_closed(self, retargeter):
        op = retargeter.update(type("Mo",(),{"landmarks_image":self._lm(0.5,0.08),"landmarks_world":self._lm(0.5,0.08)})())
        retargeter.reset_origin()
        cl = retargeter.update(type("Mo",(),{"landmarks_image":self._lm(0.5,0.005),"landmarks_world":self._lm(0.5,0.005)})())
        assert op.gripper_width > cl.gripper_width + 1e-4
        assert op.gripper_width - cl.gripper_width > 1e-3

    def test_gripper_range(self, retargeter):
        for s in [0.0,0.02,0.05,0.10,0.15]:
            P = self._lm(0.5,s)
            t = retargeter.update(type("Mo",(),{"landmarks_image":P,"landmarks_world":P})())
            retargeter.reset_origin()
            assert 0.0 <= t.gripper_width <= 0.04, f"spread={s} -> {t.gripper_width}"


class TestDepthMapping:
    def test_depth_enabled_changes_x(self):
        ret = HandToPandaRetargeter(
            robot_origin=np.array([0.45, 0.0, 0.45]),
            position_scale_xy=2.0, depth_scale=1.0,
            enable_depth_mapping=True, filter_alpha=1.0)

        P0 = _make_landmarks(palm_span=0.12)
        P1 = _make_landmarks(palm_span=0.15)  # larger palm = closer

        _ = ret.map_position_from_image(P0)   # baseline
        pos = ret.map_position_from_image(P1)
        # x should differ from origin because depth is enabled
        assert abs(pos[0] - ret.robot_origin[0]) > 1e-4, f"x={pos[0]:.4f}"

    def test_depth_disabled_keeps_x(self):
        ret = HandToPandaRetargeter(
            robot_origin=np.array([0.45, 0.0, 0.45]),
            position_scale_xy=2.0, depth_scale=1.0,
            enable_depth_mapping=False, filter_alpha=1.0)

        P0 = _make_landmarks(palm_span=0.12)
        P1 = _make_landmarks(palm_span=0.15)

        _ = ret.map_position_from_image(P0)   # baseline
        pos = ret.map_position_from_image(P1)
        # x should stay at origin because depth is disabled
        assert abs(pos[0] - ret.robot_origin[0]) < 1e-4, f"x={pos[0]:.4f}"

    def test_depth_clamped_to_workspace(self):
        ret = HandToPandaRetargeter(
            robot_origin=np.array([0.45, 0.0, 0.45]),
            position_scale_xy=2.0, depth_scale=10.0,
            enable_depth_mapping=True, filter_alpha=1.0)

        P0 = _make_landmarks(palm_span=0.12)
        P_big = _make_landmarks(palm_span=0.50)  # extreme size change

        _ = ret.map_position_from_image(P0)
        pos = ret.map_position_from_image(P_big)
        # must stay within workspace bounds
        assert ret.robot_origin[0] - 0.11 <= pos[0] <= ret.robot_origin[0] + 0.11


if __name__ == "__main__":
    pytest.main([__file__])