#!/usr/bin/env python
"""
Launch the Cosita robot real-time viewer.

Usage:
    python run_gui.py
"""

if __name__ == "__main__":
    print("EEZYbotARM Viewer")
    print("=" * 40)

    from robots.eezybotarm import EEZYbotARM
    robot = EEZYbotARM()

    from robot_viewer import launch
    launch(robot, title="EEZYbotARM Viewer")
