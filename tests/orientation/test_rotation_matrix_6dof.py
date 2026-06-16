import pytest
import numpy as np
from scipy.spatial.transform import Rotation


def test_rotation_matrix_orthogonality():
    R = Rotation.random().as_matrix()
    I_check = R.T @ R
    assert np.allclose(I_check, np.eye(3), atol=1e-10)

