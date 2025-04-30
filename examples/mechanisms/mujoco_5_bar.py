import mujoco
import pygame
import mujoco.viewer
import numpy as np

# PID parameters
Kp = 50.0  # Proportional gain
Kd = 1.0   # Derivative gain, 
qstar = np.deg2rad([-65,65])  # Target angles for the two joints
er = [0.0, 0.0]  # Previous error values for derivative calculation
kDeltaAngle=np.deg2rad(1e-3)

qpos_0=np.deg2rad([-65,-90,0,65,90])  # Initial angles for the two joints
# Init pygame joystick
pygame.init()
joystick = pygame.joystick.Joystick(0)
joystick.init()

# Load model
model = mujoco.MjModel.from_xml_path("examples/mechanisms/5bar.xml")
data = mujoco.MjData(model)
data.qpos = qpos_0  # Set initial joint angles
kActiveJointsIdx=[0,3]

# Simulation loop
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        pygame.event.pump()

        axes = [joystick.get_axis(i) for i in range(joystick.get_numaxes())]
        buttons = [joystick.get_button(i) for i in range(joystick.get_numbuttons())]
        hats = [joystick.get_hat(i) for i in range(joystick.get_numhats())]

        # Get joystick input for target angles
        if np.abs(axes[0]) > 0.1:
            qstar[0] += kDeltaAngle*-np.sign(axes[0])        
        if np.abs(axes[2]) > 0.1:
            qstar[1] += kDeltaAngle*-np.sign(axes[2])                            

        for i in range(2):  # Assuming two joints are controlled
            er_i=qstar[i]-data.qpos[kActiveJointsIdx[i]]
            erd_i=(er_i-er[i])/model.opt.timestep
            er[i] = er_i
            control_output = Kp * er_i - Kd * erd_i          
            data.ctrl[i] = control_output  # Map to joint motor input                
         
        mujoco.mj_step(model, data)
        viewer.sync()