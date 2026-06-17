"""Tests for core/operation_logger.py."""

import os
import tempfile

import numpy as np
import pytest

from core.operation_logger import OperationLogger, OperationSample, _finite


class TestFinite:
    def test_passes_on_finite(self):
        _finite(np.array([1.0, 2.0, 3.0]), "x")

    def test_raises_on_nan(self):
        with pytest.raises(ValueError, match="non-finite"):
            _finite(np.array([1.0, np.nan]), "x")

    def test_skips_none(self):
        _finite(None, "x")  # should not raise


class TestLogger:
    @pytest.fixture
    def logger(self):
        return OperationLogger()

    def make_sample(self, step=0, t=0.0, delta=None, valid=True):
        return OperationSample(
            step=step,
            time=t,
            target_pos=np.array([0.45, 0.0, 0.45]),
            action_pos_delta=delta,
            gripper_width=0.02,
            valid=valid,
            source="test",
        )

    def test_append_and_count(self, logger):
        logger.append(self.make_sample(0))
        logger.append(self.make_sample(1, 0.1))
        assert len(logger) == 2

    def test_save_csv(self, logger):
        logger.append(self.make_sample(0))
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            logger.save_csv(path)
            assert os.path.getsize(path) > 0
            with open(path, "r", encoding="utf-8") as fh:
                lines = fh.readlines()
            assert len(lines) == 2  # header + 1 data row
        finally:
            os.remove(path)

    def test_save_npz(self, logger):
        for i in range(3):
            logger.append(self.make_sample(i, i * 0.05))
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            logger.save_npz(path)
            loaded = OperationLogger.load_npz(path)
            assert loaded["step"].shape == (3,)
            assert loaded["target_pos"].shape == (3, 3)
            assert bool(np.all(np.isfinite(loaded["target_pos"])))
        finally:
            os.remove(path)

    def test_action_delta_stored(self, logger):
        delta = np.array([0.01, -0.005, 0.002])
        logger.append(self.make_sample(0, 0.0, delta=delta))
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            logger.save_csv(path)
            with open(path, "r", encoding="utf-8") as fh:
                lines = fh.readlines()
            cols = lines[1].strip().split(",")
            assert float(cols[8]) == pytest.approx(0.01)
            assert float(cols[9]) == pytest.approx(-0.005)
            assert float(cols[10]) == pytest.approx(0.002)
        finally:
            os.remove(path)

    def test_invalid_sample_valid_flag(self, logger):
        logger.append(self.make_sample(0, valid=False))
        logger.append(self.make_sample(1, valid=True))
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            logger.save_npz(path)
            loaded = OperationLogger.load_npz(path)
            assert loaded["valid"].tolist() == [False, True]
        finally:
            os.remove(path)


if __name__ == "__main__":
    pytest.main([__file__])
