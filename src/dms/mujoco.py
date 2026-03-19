import mujoco


def make_camera(model, azimuth=90, elevation=-20, distance_scale=2.0):
    """Create a free camera auto-scaled to the model geometry.

    Args:
        model: MjModel instance.
        azimuth: horizontal angle in degrees (90 = side view, 0 = front).
        elevation: vertical angle in degrees (negative = above).
        distance_scale: multiplier on model.stat.extent for camera distance.

    Returns:
        MjvCamera ready to pass to renderer.update_scene(data, camera=cam).
    """
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = model.stat.center
    cam.distance = distance_scale * model.stat.extent
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam
