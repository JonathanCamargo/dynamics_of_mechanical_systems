# Dynamics of Mechanical Systems

Course materials and tools for Dynamics of Mechanical Systems. Includes a Python library (`dms`) for symbolic and numerical analysis, along with simulation examples using MuJoCo.

## Install

Clone the repository and install with:

```bash
pip install -r requirements.txt
```

## Structure

- **src/dms/** - Core library: symbolic dynamics (SymPy), curves, mechanisms, and bundled MuJoCo tools and models.
- **examples/** - Standalone demos: MuJoCo simulations, joystick control, mechanisms, curves.
- **tutorials/** - Step-by-step notebooks covering rotations, curves, MuJoCo, and EKF.

## Quick Start

```bash
python examples/mujoco/pendulum_trajectory_control.py
```
