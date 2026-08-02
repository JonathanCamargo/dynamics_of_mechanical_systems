#!/usr/bin/env python
"""
Launch the Eezybotarm robot real-time viewer.

Usage:
    python run_gui.py
"""

if __name__ == "__main__":
    print("EEZYbotARM Viewer")
    print("=" * 40)

    from dms.robots import EEZYbotARM
    robot = EEZYbotARM()

    from dms.robots.robot_viewer import launch
    launch(robot, title="EEZYbotARM Viewer")
