import pytest

pytestmark = pytest.mark.viewer


def test_viewer_gravity_compensation():
    pytest.skip("requires viewer GUI and display")