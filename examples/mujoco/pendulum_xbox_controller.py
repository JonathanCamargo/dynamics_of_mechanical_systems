"""Control a simple pendulum with an Xbox controller.

Left stick Y-axis applies torque directly to the pendulum hinge.
Uses the pendulum1.xml model which has a torque actuator on the pin joint.
"""
import mujoco
import pygame
import time

import dsm

# Torque scaling: joystick horizontal axis [-1, 1] maps to [-MAX_TORQUE, MAX_TORQUE]
DELTA_TORQUE = 0.05
DEADZONE = 0.1

# Init pygame joystick
pygame.init()
joystick = pygame.joystick.Joystick(0)
joystick.init()
print(f"Controller: {joystick.get_name()}")

# Load pendulum model (has motor actuator "torque" on joint "pin")
model = mujoco.MjModel.from_xml_path(dsm.get_asset_path("mujoco_models/pendulum1.xml"))
data = mujoco.MjData(model)

# Start pendulum hanging down (qpos=0 is vertical in this model)
data.qpos[0] = 0.0

with mujoco.viewer.launch_passive(model, data) as viewer:
     while viewer.is_running():
        pygame.event.pump()
        # Left stick X-axis (axis 0 on Xbox controllers)        
        axis_val = joystick.get_axis(0)        
        if abs(axis_val) < DEADZONE:
            axis_val = 0.0
        # Apply torque directly
        data.ctrl[0] += DELTA_TORQUE * axis_val
        step_start = time.time()
        mujoco.mj_step(model, data)
        viewer.sync()
        elapsed = time.time() - step_start
        dt = model.opt.timestep
        if elapsed < dt:
            time.sleep(dt - elapsed)    
