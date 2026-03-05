#!/usr/bin/env python
"""
RR Robot viewer with workspace visualisation and obstacles.

Usage:
    python run_gui.py
"""

import sys
import numpy as np

if __name__ == "__main__":
    print("RR Robot - Workspace Viewer")
    print("=" * 40)

    from dms.robots import RR
    from dms.robots.workspace import (
        compute_workspace,
        add_obstacles,
        check_collision,
        CircleObstacle,
        PolygonObstacle,
    )

    robot = RR()
    robot._limits = (np.deg2rad([45, -60]), np.deg2rad([135, 90]))

    # ── Compute workspace ─────────────────────────────────────────────
    print("  Computing workspace ...")
    ws_x, ws_y = compute_workspace(robot, n=200)
    print(f"  {len(ws_x)} reachable points computed.")

    # ── Define obstacles ──────────────────────────────────────────────
    obstacles = [
        CircleObstacle(1.0, 1.0, 0.3),
        PolygonObstacle([(-0.5, 0.8), (-0.2, 1.3), (-0.8, 1.3)]),
    ]

    # ── Launch viewer ─────────────────────────────────────────────────
    from PyQt5.QtWidgets import QApplication, QGroupBox, QVBoxLayout, QLabel
    from PyQt5.QtGui import QFont
    from dms.robots.robot_viewer.gui import RobotViewer, DARK_STYLE

    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLE)

    win = RobotViewer(robot, title="RR Robot - Workspace Viewer")

    # Workspace scatter (drawn once, becomes part of the cached background)
    win.canvas.ax.scatter(
        ws_x, ws_y, s=0.1, c="#89b4fa", alpha=0.15, zorder=0,
    )

    # Obstacles
    add_obstacles(win.canvas.ax, obstacles)

    # ── Collision indicator ───────────────────────────────────────────
    # Add a "Collision" status box to the right-hand control panel.
    from PyQt5.QtWidgets import QSplitter

    splitter = win.centralWidget().findChild(QSplitter)
    ctrl_panel = splitter.widget(1)
    ctrl_layout = ctrl_panel.layout()

    collision_box = QGroupBox("Collision")
    coll_lay = QVBoxLayout(collision_box)
    collision_label = QLabel("Clear")
    collision_label.setFont(QFont("Consolas", 12))
    collision_label.setStyleSheet("color: #a6e3a1; font-weight: bold;")
    coll_lay.addWidget(collision_label)
    # Insert before the trailing stretch
    ctrl_layout.insertWidget(ctrl_layout.count() - 1, collision_box)

    # Patch _update_fk to check collisions after every pose update
    _original_update_fk = win._update_fk

    def _update_fk_with_collision():
        _original_update_fk()
        if win._last_eef:
            if check_collision(*win._last_eef, obstacles):
                collision_label.setText("COLLISION!")
                collision_label.setStyleSheet(
                    "color: #f38ba8; font-weight: bold;"
                )
            else:
                collision_label.setText("Clear")
                collision_label.setStyleSheet(
                    "color: #a6e3a1; font-weight: bold;"
                )

    win._update_fk = _update_fk_with_collision

    win.show()
    sys.exit(app.exec_())
