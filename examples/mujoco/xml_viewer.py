"""Load a MuJoCo XML model and open the interactive viewer.

Accepts a file path or a dms asset name (e.g. "pendulum").

Usage:
    python xml_viewer.py pendulum
    python xml_viewer.py pendulum --qpos 1.57
    python xml_viewer.py double_pendulum --qpos 1.57 0.5 --qvel 0 1.0
    python xml_viewer.py path/to/model.xml
    python xml_viewer.py model.xml --key 0
"""
import argparse
from pathlib import Path

import dms
import mujoco
import mujoco.viewer


def resolve_model_path(name):
    """Resolve a file path or dms asset name to an XML path."""
    if Path(name).is_file():
        return str(name)
    return str(dms.get_asset_path("mujoco_models/" + name + ".xml"))


def main():
    parser = argparse.ArgumentParser(description="Load and simulate a MuJoCo XML model.")
    parser.add_argument("model", help="Path to XML file or dms asset name (e.g. 'pendulum')")
    parser.add_argument("--qpos", type=float, nargs="+", help="Initial joint positions")
    parser.add_argument("--qvel", type=float, nargs="+", help="Initial joint velocities")
    parser.add_argument("--key", type=int, default=-1,
                        help="Keyframe index to load (default: -1 for none)")
    args = parser.parse_args()

    model_path = resolve_model_path(args.model)
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Initial state: keyframe takes precedence, then --qpos/--qvel
    if args.key >= 0 and model.nkey > args.key:
        mujoco.mj_resetDataKeyframe(model, data, args.key)
    if args.qpos is not None:
        data.qpos[:len(args.qpos)] = args.qpos
    if args.qvel is not None:
        data.qvel[:len(args.qvel)] = args.qvel
    mujoco.mj_forward(model, data)

    # launch() runs the viewer in the main thread — UI panels render correctly
    # on Windows (launch_passive runs in a separate thread and can have UI issues).
    # Built-in controls: space=pause, right arrow=step, mouse=camera, UI panels.
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
