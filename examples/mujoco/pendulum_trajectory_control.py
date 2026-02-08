# Computed torque control: make a pendulum track a sinusoidal trajectory.
# Uses inverse dynamics to compute the feedforward torque, then compares
# the simulated trajectory against the analytical reference.
#
# Usage:
#   python pendulum_trajectory_control.py

import math
import time
import numpy as np
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import dsm

# ========================
# MODEL
# ========================
model = mujoco.MjModel.from_xml_path(dsm.get_asset_path("mujoco_models/pendulum1.xml"))
data = mujoco.MjData(model)

# ========================
# LOAD SYSTEM PARAMETERS FROM MODEL
# ========================

m = model.body_mass[1]                          # mass [kg]
L = 2 * model.geom_size[1, 1]                  # rod length [m] (full length of cylinder geom)
g = abs(model.opt.gravity[2])                   # gravity magnitude [m/s^2]
d = model.jnt_pos[0, 2]                         # joint offset from body COM [m]
I_com = model.body_inertia[1, 1]                # inertia about COM, Y-axis (hinge axis)
I = I_com + m * d ** 2                           # inertia about pivot (parallel axis theorem)
c = model.dof_damping[0]                         # damping coefficient

# ========================
# MANUAL SYSTEM PARAMETERS OVERRIDE (OPTIONAL)
# ========================
OVERRIDE_MODEL_PARAMS = False
if OVERRIDE_MODEL_PARAMS:
    m = 1.0                    # mass [kg]
    L = 1.0                    # rod length [m]
    g = 9.81                   # gravity [m/s^2]
    I = (1 / 3) * m * L ** 2  # inertia of rod pivoted at one end
    c = 1.0                    # damping coefficient

# ========================
# DESIRED TRAJECTORY
# ========================
theta_max = math.radians(45)       # amplitude: +/-45 deg
f = 0.5                            # frequency [Hz]
omega = 2 * math.pi * f            # angular frequency [rad/s]
duration = 10                      # simulation duration [s]

# ========================
# CONTROLLER
# ========================
def controller(model, data):
    t = data.time
    theta_d = theta_max * math.sin(omega * t)
    theta_dot_d = theta_max * omega * math.cos(omega * t)
    theta_ddot_d = -theta_max * omega ** 2 * math.sin(omega * t)

    torque_ff = I * theta_ddot_d + c * theta_dot_d + m * g * (L / 2) * math.sin(theta_d)
    data.ctrl[0] = torque_ff

# ========================
# SIMULATION
# ========================
mujoco.mj_resetData(model, data)
data.qpos[0] = 0.0
data.qvel[0] = theta_max * omega  # match initial velocity of the reference
mujoco.mj_forward(model, data)

q_log, t_log = [], []

try:
    mujoco.set_mjcb_control(controller)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "side")
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

        while viewer.is_running() and data.time < duration:
            step_start = time.time()
            mujoco.mj_step(model, data)
            q_log.append(data.qpos[0].copy())
            t_log.append(data.time)
            viewer.sync()
            elapsed = time.time() - step_start
            dt = model.opt.timestep
            if elapsed < dt:
                time.sleep(dt - elapsed)
finally:
    mujoco.set_mjcb_control(None)

# ========================
# RESULTS
# ========================
q_log = np.array(q_log)
t_log = np.array(t_log)
q_ref = theta_max * np.sin(omega * t_log)

print(f"Max angle reached: {np.degrees(np.max(q_log)):.2f} deg")
print(f"Min angle reached: {np.degrees(np.min(q_log)):.2f} deg")

plt.figure(figsize=(10, 4))
plt.plot(t_log, np.degrees(q_log), label="Executed Trajectory")
plt.plot(t_log, np.degrees(q_ref), "--", label="Reference (+/-45 deg)")
plt.xlabel("Time [s]")
plt.ylabel("Angle [deg]")
plt.title("Pendulum Trajectory Tracking (Computed Torque)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
