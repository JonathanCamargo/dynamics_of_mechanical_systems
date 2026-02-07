"""Setup verification tests.

Run with: pytest tests/test_setup.py -v

These tests verify that all dependencies are installed correctly
and that the MuJoCo pendulum model can be loaded, simulated, and rendered.
"""

from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "models" / "pendulum.xml"


def test_imports():
    """Verify all required packages can be imported."""
    import mujoco
    import numpy
    import matplotlib
    import mediapy
    import ipywidgets
    import scipy

    # Ensure they are not None (actually loaded)
    assert mujoco is not None
    assert numpy is not None
    assert matplotlib is not None
    assert mediapy is not None
    assert ipywidgets is not None
    assert scipy is not None


def test_mujoco_model():
    """Verify MuJoCo can load the pendulum model with correct structure."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))

    # Pendulum has 1 joint (hinge)
    assert model.nq == 1, f"Expected 1 generalized coordinate, got {model.nq}"
    assert model.nv == 1, f"Expected 1 velocity, got {model.nv}"

    # Model has 1 sensor (joint position)
    assert model.nsensor == 1, f"Expected 1 sensor, got {model.nsensor}"


def test_mujoco_simulation():
    """Verify MuJoCo can simulate the pendulum (100 steps)."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    # Set initial angle so pendulum swings
    data.qpos[0] = 0.5  # radians

    # Run 100 simulation steps
    for _ in range(100):
        mujoco.mj_step(model, data)

    # Time should have advanced
    assert data.time > 0, "Simulation time did not advance"
    # With initial angle and gravity, pendulum should have moved
    assert data.qpos[0] != 0.5, "Pendulum position did not change"


def test_mujoco_renderer():
    """Verify MuJoCo Renderer can produce a frame."""
    import mujoco
    import numpy as np

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)

    # Step once so there is state to render
    mujoco.mj_step(model, data)

    # Render a frame
    renderer.update_scene(data)
    frame = renderer.render()

    # Verify frame shape: (height, width, 3) RGB
    assert frame.shape == (480, 640, 3), f"Expected (480, 640, 3), got {frame.shape}"
    assert frame.dtype == np.uint8, f"Expected uint8, got {frame.dtype}"

    renderer.close()
