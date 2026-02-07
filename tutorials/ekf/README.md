# EKF MuJoCo Tutorial

An interactive Jupyter notebook tutorial teaching Extended Kalman Filters (EKF) through hands-on demos with MuJoCo-simulated mechanical systems. Engineers new to state estimation learn to implement, apply, and debug EKFs by working through progressively complex physical systems -- from a simple pendulum to multi-body robots.

## Learning Objectives

By the end of this tutorial, you will be able to:

- Implement an EKF from scratch using NumPy (predict/update loop, Jacobians, covariance propagation)
- Apply EKF to mechanical systems (pendulum, cart-pole, double pendulum, robot arm)
- Tune Q/R covariance matrices using physical intuition rather than guesswork
- Diagnose EKF failures and divergence (and know how to fix them)

## Prerequisites

- Python 3.9+
- ffmpeg (for video display in notebooks) -- install via your system package manager
- Basic linear algebra and dynamics background

## Installation

```bash
git clone https://github.com/your-username/mujoco_explained.git
cd mujoco_explained
pip install -r requirements.txt
```

## Quick Start

```bash
jupyter lab
```

Then open `notebooks/01_pendulum_ekf.ipynb` and follow along.

## Notebook Overview

| Notebook | System | Concepts |
|----------|--------|----------|
| 01 - Pendulum EKF | Simple pendulum | EKF from scratch, Jacobians, Q/R tuning, position-only sensing |
| 02 - Cart-Pole EKF | Cart-pole | Multi-state EKF, partial observability |
| 03 - Advanced Systems | Double pendulum, robot arm | Sensor fusion, multi-rate sensors |
| 04 - Robustness | All systems | Sensor failures, divergence diagnosis |

## Project Structure

```
mujoco_explained/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── models/
│   └── pendulum.xml             # MuJoCo pendulum model
├── notebooks/
│   └── 01_pendulum_ekf.ipynb    # EKF pendulum tutorial
└── tests/
    └── test_setup.py            # Environment verification
```

## Verification

After installation, verify your setup:

```bash
pytest tests/test_setup.py -v
```

All 4 tests should pass, confirming MuJoCo, NumPy, and visualization dependencies are working.
