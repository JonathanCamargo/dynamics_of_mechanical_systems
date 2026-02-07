import sys
import mujoco
import mujoco.viewer

model_path = sys.argv[1]
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Use next() with generator expression and default to None for a clean search
camera_name = "side"
cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)

with mujoco.viewer.launch_passive(model, data) as viewer:
    # Set the camera if it exists
    if cam_id != -1:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = cam_id
    else:
        # Fallback: set up a free camera with custom parameters
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -20
        viewer.cam.distance = 3.0
        viewer.cam.lookat[:] = [0, 0, 0.5]

    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()