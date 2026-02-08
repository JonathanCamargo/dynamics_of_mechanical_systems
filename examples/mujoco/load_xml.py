# Load a MuJoCo XML model and run a real-time simulation in the viewer.
# Accepts a file path or a dms asset name (e.g. "pendulum").
#
# Usage:
#   python load_xml.py pendulum
#   python load_xml.py pendulum --qpos 1.57
#   python load_xml.py double_pendulum --qpos 1.57 0.5 --qvel 0 1.0
#   python load_xml.py path/to/model.xml

import argparse
from pathlib import Path
import dms
import mujoco
import mujoco.viewer
import time

parser = argparse.ArgumentParser(description="Load and simulate a MuJoCo XML model.")
parser.add_argument("model", help="Path to XML file or dms asset name (e.g. 'pendulum')")
parser.add_argument("--qpos", type=float, nargs="+", help="Initial joint positions")
parser.add_argument("--qvel", type=float, nargs="+", help="Initial joint velocities")
args = parser.parse_args()

if Path(args.model).is_file():
    model_path = args.model
else:
    model_path = dms.get_asset_path("mujoco_models/" + args.model + ".xml")
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

if args.qpos is not None:
    data.qpos[:len(args.qpos)] = args.qpos
if args.qvel is not None:
    data.qvel[:len(args.qvel)] = args.qvel

# Load a camera by name if it exists
camera_name = "side"
cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()
    if cam_id != -1:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = cam_id
    else:
        # Auto-frame: use model stats to fit the camera to the scene
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = model.stat.center
        viewer.cam.distance = 2.0 * model.stat.extent
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -20

    while viewer.is_running():        
        step_start = time.time()
        mujoco.mj_step(model, data)
        viewer.sync()
        # Sleep until the next timestep
        elapsed = time.time() - step_start
        dt = model.opt.timestep
        if elapsed < dt:
            time.sleep(dt - elapsed)