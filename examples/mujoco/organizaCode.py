import os
import math
import numpy as np
import matplotlib.pyplot as plt
import mediapy as media
import mujoco

# ========================
# CONFIGURACIÓN DEL MODELO
# ========================
xml_path = "pendulum1.xml"
dirname = os.path.dirname(__file__)
abs_path = os.path.join(dirname, xml_path)

model = mujoco.MjModel.from_xml_path(abs_path)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)

# ========================
# PARÁMETROS DEL SISTEMA
# ========================
m = 1.0        # masa [kg]
L = 1.0        # longitud de la barra [m]
g = 9.81       # gravedad [m/s^2]
I = (1/3) * m * L**2   # inercia de barra pivotada en el extremo
c = 1        # amortiguamiento
# ========================
# TRAYECTORIA DESEADA
# ========================
theta_max = math.radians(45)   # amplitud de ±45°
f = 0.5                        # frecuencia [Hz]
omega = 2 * math.pi * f        # frecuencia angular [rad/s]
duration = 5                   # duración de la simulación [s]
framerate = 30                 # cuadros por segundo para el video

# ========================
# CONTROLADOR
# ========================
def my_controller(model, data):
    t = data.time
    
    # Trayectoria analítica
    theta = theta_max * math.sin(omega * t)               # posición deseada
    theta_dot = theta_max * omega * math.cos(omega * t)   # velocidad deseada
    theta_ddot = -theta_max * omega**2 * math.sin(omega * t)  # aceleración deseada
    
    # Torque analítico (dinámica inversa)
    torque_ff = I * theta_ddot + c * theta_dot+ m * g * (L/2) * np.sin(theta)
    
    data.ctrl = [torque_ff]
    return

# ========================
# SIMULACIÓN
# ========================
mujoco.mj_resetData(model, data)

data.qpos[0] = 0.0  # Ángulo inicial [rad]
data.qvel[0] = theta_max * omega # Velocidad angular inicial [rad/s]
mujoco.mj_forward(model, data)

q, w, a, t, frames = [], [], [], [], []

try:
    mujoco.set_mjcb_control(my_controller) 
    while data.time < duration: 
        mujoco.mj_step(model, data) 
        q.append(data.qpos.copy())  
        w.append(data.qvel.copy())
        a.append(data.qacc.copy())
        t.append(data.time) 

        if len(frames) < data.time * framerate:
            renderer.update_scene(data)
            pixels = renderer.render()
            frames.append(pixels)
finally:
    mujoco.set_mjcb_control(None)
    renderer.close()

# ========================
# VIDEO
# ========================
media.write_video("pendulum_45deg_Test.mp4", frames, fps=framerate)
print("Video guardado como pendulum_45deg_controlled.mp4")

# ========================
# RESULTADOS
# ========================
q = np.array(q).flatten() 
w = np.array(w).flatten() 
a = np.array(a).flatten()
t = np.array(t)

print(f"Ángulo máximo alcanzado: {np.degrees(np.max(q)):.2f}°")
print(f"Ángulo mínimo alcanzado: {np.degrees(np.min(q)):.2f}°")

plt.figure(figsize=(9,4))
plt.plot(t, np.degrees(q), label="Simulado (MuJoCo)")
plt.plot(t, np.degrees(theta_max * np.sin(omega * t)), "--", label="Analítico (±45°)")
plt.xlabel("Tiempo [s]")
plt.ylabel("Ángulo [°]")
plt.title("Oscilación controlada del péndulo entre ±45°")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()