#!/usr/bin/env python
"""
Fine-grained diagnostic: isolates the matplotlib Qt5Agg backend import
(the suspected crash site) from the dms imports, printing before each step
so the last line printed tells us exactly what faulted.

Run:  python _diag_gui.py
"""

import sys, faulthandler
faulthandler.enable()

def step(msg):
    print(msg, flush=True)

step("[1] start")

step("[2] import PyQt5.QtWidgets ...")
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

step("[3] import matplotlib + use('Qt5Agg') ...")
import matplotlib
matplotlib.use("Qt5Agg")

step("[4] import matplotlib.pyplot ...")
import matplotlib.pyplot as plt

step("[5] import backend_qt5agg (FigureCanvasQTAgg) ...")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

step("[6] create QApplication ...")
app = QApplication.instance() or QApplication(sys.argv)

step("[7] create a bare FigureCanvas ...")
canvas = FigureCanvasQTAgg(Figure())

step("[8] import dms.robots.rrr ...")
from dms.robots import rrr

step("[9] build robot ...")
robot = rrr()

step("[10] import + build RobotViewer ...")
from dms.robots.robot_viewer.gui import RobotViewer
win = RobotViewer(robot, title="RRR diag")

step("[11] show() ...")
win.show()
step(f"[11b] visible={win.isVisible()}")

QTimer.singleShot(2000, app.quit)
rc = app.exec_()
step(f"[12] event loop returned: {rc}")
