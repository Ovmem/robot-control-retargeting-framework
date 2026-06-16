# Testing Strategy

The project uses pytest for test discovery and execution.

## Automated tests

Automated tests are plain pytest tests that:

- Do not require a display, viewer GUI, or camera
- Can run headless (no MuJoCo viewer)
- Complete within seconds
- Have explicit assert statements

Run automated tests (skips viewer/interactive tests):

`ash
pytest -q -m "not viewer and not interactive"
`

To also exclude MuJoCo model loading tests:

`ash
pytest -q -m "not viewer and not interactive and not mujoco"
`

Run only MuJoCo-related tests:

`ash
pytest -q -m mujoco
`

## Interactive / Viewer validation

Tests marked with \@pytest.mark.viewer\ require:

- A display and MuJoCo GUI viewer
- Manual visual inspection
- Usually involve \mujoco.viewer.launch_passive()\

These tests are **skipped by default** in \pytest\ or \pytest -m "not viewer and not interactive"\.

They can still be run manually as standalone scripts:

`ash
python tests/panda/test_view_panda.py
python tests/panda/test_gravity_compensation.py
python tests/panda/test_pd_control.py
python tests/panda/test_pd_gravity_compensation.py
`

## Hand retargeting demo

\demos/panda/demo_hand_retargeting_pd_gc.py\ is not a pytest test. It requires:

- A working camera
- OpenCV GUI window
- MuJoCo viewer display

## Custom markers

Registered in \pytest.ini\:

| Marker | Description |
|---|---|
| \interactive\ | Requires viewer, GUI, camera, or manual inspection |
| \iewer\ | Requires MuJoCo viewer and display |
| \mujoco\ | Requires MuJoCo model loading |
| \slow\ | Longer-running validation |
