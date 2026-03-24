"""
Real-time robot mechanism viewer.

Usage::

    from dms.robots.robot_viewer import launch
    launch(my_robot, title="My Robot")
"""

import sys
import numpy as np
from collections import deque

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.backends.backend_qt5agg import (      # noqa: E402
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.figure import Figure                   # noqa: E402

from PyQt5.QtWidgets import (                          # noqa: E402
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QCheckBox, QButtonGroup,
    QSplitter, QGroupBox, QSizePolicy, QComboBox,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal         # noqa: E402
from PyQt5.QtGui import QFont                          # noqa: E402

from ..model import RobotModel                           # noqa: E402

# ── Dark theme stylesheet ────────────────────────────────────────────────────

DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: "Segoe UI", "Ubuntu", sans-serif;
    font-size: 13px;
}
QGroupBox {
    border: 1px solid #45475a;
    border-radius: 8px;
    margin-top: 14px;
    padding: 14px 8px 8px 8px;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #45475a;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #89b4fa;
    width: 16px; height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}
QSlider::handle:horizontal:hover {
    background: #b4d0fb;
}
QSlider::handle:horizontal:disabled {
    background: #585b70;
}
QPushButton {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 7px 14px;
    font-weight: bold;
}
QPushButton:hover { background-color: #45475a; }
QPushButton:checked {
    background-color: #89b4fa;
    color: #1e1e2e;
    border-color: #89b4fa;
}
QCheckBox { spacing: 8px; }
QCheckBox::indicator {
    width: 16px; height: 16px;
    border-radius: 4px;
    border: 2px solid #45475a;
    background: #313244;
}
QCheckBox::indicator:checked {
    background: #89b4fa;
    border-color: #89b4fa;
}
QLabel { background: transparent; }
"""

# ── Matplotlib colours for dark background ───────────────────────────────────

_MPL_RC = {
    "figure.facecolor": "#1e1e2e",
    "axes.facecolor": "#181825",
    "axes.edgecolor": "#45475a",
    "axes.labelcolor": "#a6adc8",
    "xtick.color": "#6c7086",
    "ytick.color": "#6c7086",
    "text.color": "#cdd6f4",
    "grid.color": "#45475a",
    "grid.alpha": 0.25,
}

# ── Canvas ───────────────────────────────────────────────────────────────────

class RobotCanvas(FigureCanvas):
    """Matplotlib canvas that draws the robot and emits click/drag signals.

    Uses blitting for fast redraws: the static background (axes, grid, ticks)
    is rasterized once and cached; each frame only the dynamic artists are
    redrawn on top of that cached image.
    """

    ik_requested = pyqtSignal(float, float)

    def __init__(self, model: RobotModel, parent=None):
        for k, v in _MPL_RC.items():
            matplotlib.rcParams[k] = v

        self.fig = Figure(tight_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)
        self.model = model

        self.ax = self.fig.add_subplot(111)
        xlim, ylim = model.view_limits()
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.ax.set_aspect("equal")
        self.ax.grid(True)

        # Pre-create artists for fast updates (animated=True excludes
        # them from the normal full-figure draw so they don't burn into
        # the cached background).
        self._animated_artists = []

        self._link_artists = {}
        for lk in model.get_links():
            key = f"{lk.start}->{lk.end}"
            (line,) = self.ax.plot(
                [], [],
                color=lk.color,
                linewidth=lk.linewidth,
                solid_capstyle="round",
                animated=True,
            )
            self._link_artists[key] = line
            self._animated_artists.append(line)

        self._point_artists = {}
        for pt in model.get_points():
            (marker,) = self.ax.plot(
                [], [],
                marker=pt.marker,
                color=pt.color,
                markersize=pt.size,
                linestyle="none",
                animated=True,
            )
            self._point_artists[pt.name] = marker
            self._animated_artists.append(marker)

        # IK target cross-hair
        (self._target,) = self.ax.plot(
            [], [], "+", color="#f38ba8", markersize=18, markeredgewidth=1.5,
            animated=True,
        )
        self._animated_artists.append(self._target)

        # EEF trail
        self._trail = deque(maxlen=800)
        self._show_trail = False
        (self._trail_line,) = self.ax.plot(
            [], [], color="#f38ba8", alpha=0.45, linewidth=1.2,
            animated=True,
        )
        self._animated_artists.append(self._trail_line)

        # Blitting state -- background captured after first full draw
        self._bg = None
        self.mpl_connect("draw_event", self._on_draw)

        # Mouse interaction state
        self._ik_active = False
        self._dragging = False
        self.mpl_connect("button_press_event", self._on_press)
        self.mpl_connect("motion_notify_event", self._on_motion)
        self.mpl_connect("button_release_event", self._on_release)

    # -- blitting ---------------------------------------------------------------

    def _on_draw(self, event):
        """Called after every full figure draw (initial render + resize).
        Capture the static background, then redraw animated artists so
        they stay visible after a resize / maximize."""
        self._bg = self.copy_from_bbox(self.ax.bbox)
        for artist in self._animated_artists:
            self.ax.draw_artist(artist)
        self.blit(self.ax.bbox)

    def _blit(self):
        """Fast redraw: restore cached background, redraw only the animated
        artists, then blit the result to screen."""
        if self._bg is None:
            # First frame -- full draw to prime the background, then
            # fall through to blit the artists on top.
            self.draw()
        self.restore_region(self._bg)
        for artist in self._animated_artists:
            self.ax.draw_artist(artist)
        self.blit(self.ax.bbox)

    # -- interaction ------------------------------------------------------------

    def set_ik_mode(self, active: bool):
        self._ik_active = active
        if not active:
            self._target.set_data([], [])
            self._blit()

    def _on_press(self, event):
        if self._ik_active and event.inaxes == self.ax and event.button == 1:
            self._dragging = True
            self._target.set_data([event.xdata], [event.ydata])
            self.ik_requested.emit(event.xdata, event.ydata)

    def _on_motion(self, event):
        if self._dragging and event.inaxes == self.ax:
            self._target.set_data([event.xdata], [event.ydata])
            self.ik_requested.emit(event.xdata, event.ydata)

    def _on_release(self, event):
        self._dragging = False

    # -- drawing ----------------------------------------------------------------

    def update_robot(self, points: dict[str, tuple[float, float]]):
        for lk in self.model.get_links():
            key = f"{lk.start}->{lk.end}"
            if lk.start in points and lk.end in points:
                p1, p2 = points[lk.start], points[lk.end]
                self._link_artists[key].set_data(
                    [p1[0], p2[0]], [p1[1], p2[1]]
                )

        for pt in self.model.get_points():
            if pt.name in points:
                p = points[pt.name]
                self._point_artists[pt.name].set_data([p[0]], [p[1]])

        # trail
        eef_name = self.model.eef_name
        if eef_name in points:
            self._trail.append(points[eef_name])
        if self._show_trail and self._trail:
            xs = [p[0] for p in self._trail]
            ys = [p[1] for p in self._trail]
            self._trail_line.set_data(xs, ys)
        else:
            self._trail_line.set_data([], [])

        self._blit()

    def set_show_trail(self, show: bool):
        self._show_trail = show
        if not show:
            self._trail_line.set_data([], [])
            self._blit()

    def clear_trail(self):
        self._trail.clear()
        self._trail_line.set_data([], [])
        self._blit()


# ── Joint slider widget ──────────────────────────────────────────────────────

class JointSlider(QWidget):
    """Slider for a single joint, with label and value readout."""

    value_changed = pyqtSignal(float)  # degrees

    def __init__(self, joint_info, parent=None):
        super().__init__(parent)
        self.joint = joint_info

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 4)

        header = QHBoxLayout()
        self.name_label = QLabel(joint_info.name)
        self.name_label.setStyleSheet("font-weight: bold;")
        self.val_label = QLabel(f"{joint_info.default_deg:.1f} deg")
        self.val_label.setAlignment(Qt.AlignRight)
        mono = QFont("Consolas", 12)
        self.val_label.setFont(mono)
        header.addWidget(self.name_label)
        header.addWidget(self.val_label)
        layout.addLayout(header)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(int(joint_info.min_deg * 10))
        self.slider.setMaximum(int(joint_info.max_deg * 10))
        self.slider.setValue(int(joint_info.default_deg * 10))
        self.slider.valueChanged.connect(self._on_change)
        layout.addWidget(self.slider)

    def _on_change(self, val):
        deg = val / 10.0
        self.val_label.setText(f"{deg:.1f} deg")
        self.value_changed.emit(deg)

    def set_value(self, deg: float):
        self.slider.blockSignals(True)
        self.slider.setValue(int(round(deg * 10)))
        self.val_label.setText(f"{deg:.1f} deg")
        self.slider.blockSignals(False)

    def get_deg(self) -> float:
        return self.slider.value() / 10.0


# ── Main window ──────────────────────────────────────────────────────────────

class RobotViewer(QMainWindow):
    def __init__(self, model: RobotModel, title: str = "Robot Viewer"):
        super().__init__()
        self.model = model
        self._last_eef = None
        self._gamepad = None
        self._serial = None
        self.setWindowTitle(title)
        self.resize(1050, 660)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # Left: canvas
        self.canvas = RobotCanvas(model)
        self.canvas.setMinimumWidth(500)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.ik_requested.connect(self._on_ik_request)
        splitter.addWidget(self.canvas)

        # Right: controls
        ctrl = QWidget()
        ctrl.setMaximumWidth(300)
        ctrl.setMinimumWidth(240)
        ctrl_layout = QVBoxLayout(ctrl)
        ctrl_layout.setContentsMargins(4, 4, 4, 4)
        splitter.addWidget(ctrl)

        # ── Mode selector ─────────────────────────────────
        mode_group_box = QGroupBox("Mode")
        mode_lay = QHBoxLayout(mode_group_box)
        self.fk_btn = QPushButton("FK")
        self.fk_btn.setCheckable(True)
        self.fk_btn.setChecked(True)
        self.ik_btn = QPushButton("IK")
        self.ik_btn.setCheckable(True)
        btn_group = QButtonGroup(self)
        btn_group.addButton(self.fk_btn)
        btn_group.addButton(self.ik_btn)
        btn_group.setExclusive(True)
        self.fk_btn.toggled.connect(self._mode_changed)
        mode_lay.addWidget(self.fk_btn)
        mode_lay.addWidget(self.ik_btn)
        ctrl_layout.addWidget(mode_group_box)

        # ── Joint sliders ─────────────────────────────────
        joints_box = QGroupBox("Joints")
        joints_lay = QVBoxLayout(joints_box)
        self.sliders: list[JointSlider] = []
        for ji in model.joint_info():
            s = JointSlider(ji)
            s.value_changed.connect(self._on_slider)
            self.sliders.append(s)
            joints_lay.addWidget(s)
        ctrl_layout.addWidget(joints_box)

        # ── End-effector info ─────────────────────────────
        info_box = QGroupBox("End Effector")
        info_lay = QVBoxLayout(info_box)
        mono = QFont("Consolas", 12)
        self.eef_x_label = QLabel("X:  --- mm")
        self.eef_y_label = QLabel("Y:  --- mm")
        self.eef_x_label.setFont(mono)
        self.eef_y_label.setFont(mono)
        info_lay.addWidget(self.eef_x_label)
        info_lay.addWidget(self.eef_y_label)
        ctrl_layout.addWidget(info_box)

        # ── Trail controls ────────────────────────────────
        trail_box = QGroupBox("Trail")
        trail_lay = QHBoxLayout(trail_box)
        self.trail_cb = QCheckBox("Show")
        self.trail_cb.toggled.connect(self.canvas.set_show_trail)
        trail_lay.addWidget(self.trail_cb)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.canvas.clear_trail)
        trail_lay.addWidget(clear_btn)
        ctrl_layout.addWidget(trail_box)

        # ── Reset button ──────────────────────────────────
        reset_btn = QPushButton("Reset Pose")
        reset_btn.clicked.connect(self._reset)
        ctrl_layout.addWidget(reset_btn)

        # ── Serial (optional) ─────────────────────────────
        self._init_serial(ctrl_layout)

        # ── Actions (optional) ────────────────────────────
        self._init_actions(ctrl_layout)

        # ── Gamepad (optional) ────────────────────────────
        self._init_gamepad(ctrl_layout)

        ctrl_layout.addStretch()

        # Splitter proportions
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        # Initial draw -- deferred so the canvas has a valid pixel buffer
        QTimer.singleShot(0, self._update_fk)

    # ── serial setup -----------------------------------------------------------

    def _init_serial(self, parent_layout):
        if not self.model.serial_config():
            return
        try:
            import serial                                   # noqa: F811
            import serial.tools.list_ports                  # noqa: F811
            self._serial_mod = serial
        except ImportError:
            return

        box = QGroupBox("Serial")
        lay = QVBoxLayout(box)

        row = QHBoxLayout()
        self._port_combo = QComboBox()
        self._port_combo.setMinimumWidth(100)
        row.addWidget(self._port_combo, stretch=1)
        refresh_btn = QPushButton("Scan")
        refresh_btn.setFixedWidth(50)
        refresh_btn.clicked.connect(self._serial_refresh)
        row.addWidget(refresh_btn)
        lay.addLayout(row)

        self._serial_btn = QPushButton("Connect")
        self._serial_btn.clicked.connect(self._serial_toggle)
        lay.addWidget(self._serial_btn)

        self._serial_status = QLabel("Disconnected")
        self._serial_status.setStyleSheet("color: #6c7086;")
        lay.addWidget(self._serial_status)

        parent_layout.addWidget(box)
        self._serial_refresh()

        # Send timer -- 50 Hz
        self._servo_timer = QTimer(self)
        self._servo_timer.timeout.connect(self._serial_tick)
        self._servo_timer.start(20)  # 50 Hz

    def _serial_refresh(self):
        if not hasattr(self, "_port_combo"):
            return
        from serial.tools.list_ports import comports
        self._port_combo.clear()
        for p in comports():
            self._port_combo.addItem(p.device, p.description)

    def _serial_toggle(self):
        if self._serial and self._serial.is_open:
            self._serial.close()
            self._serial = None
            self._serial_btn.setText("Connect")
            self._serial_status.setText("Disconnected")
            self._serial_status.setStyleSheet("color: #6c7086;")
            self._port_combo.setEnabled(True)
        else:
            port = self._port_combo.currentText()
            if not port:
                return
            cfg = self.model.serial_config()
            try:
                self._serial = self._serial_mod.Serial(
                    port,
                    baudrate=cfg.get("baudrate", 115200),
                    write_timeout=0.05,
                )
                self._serial_btn.setText("Disconnect")
                self._serial_status.setText(port)
                self._serial_status.setStyleSheet("color: #a6e3a1;")
                self._port_combo.setEnabled(False)
            except Exception as exc:
                self._serial_status.setText(str(exc)[:40])
                self._serial_status.setStyleSheet("color: #f38ba8;")

    def _serial_tick(self):
        """Called at 50 Hz -- sends current target to servo."""
        if not (self._serial and self._serial.is_open):
            return

        target = [s.get_deg() for s in self.sliders]
        cmd = self.model.serial_command(target)
        if cmd is None:
            return
        if cmd == getattr(self, "_serial_last_cmd", None):
            return  # identical command -- don't resend
        self._serial_last_cmd = cmd
        try:
            self._serial.reset_output_buffer()  # discard stale queued commands
            self._serial.reset_input_buffer()   # discard any unread responses
            self._serial.write(cmd)
        except self._serial_mod.SerialTimeoutException:
            pass
        except self._serial_mod.SerialException:
            self._serial_toggle()

    # ── actions setup ----------------------------------------------------------

    def _init_actions(self, parent_layout):
        acts = self.model.actions()
        s_acts = self.model.serial_actions()
        if not acts and not s_acts:
            return

        box = QGroupBox("Actions")
        lay = QVBoxLayout(box)

        for act in acts:
            btn = QPushButton(act.name)
            btn.clicked.connect(lambda _, a=act: self._start_action(a))
            lay.addWidget(btn)

        for sa in s_acts:
            btn = QPushButton(sa.name)
            btn.clicked.connect(lambda _, cmd=sa.command: self._send_action(cmd))
            lay.addWidget(btn)

        parent_layout.addWidget(box)

        # Interpolation state for waypoint actions
        self._action_timer = QTimer(self)
        self._action_timer.timeout.connect(self._action_tick)
        self._action_wps = []
        self._action_idx = 0
        self._action_progress = 0.0
        self._action_dt = 0.0

        # Name → action lookup for gamepad button_map
        self._action_lookup = {a.name: a for a in acts}
        self._serial_action_lookup = {sa.name: sa for sa in s_acts}

    def _start_action(self, action):
        """Begin interpolating through the action's waypoints."""
        current = [s.get_deg() for s in self.sliders]
        self._action_wps = [current] + [wp[:] for wp in action.waypoints]
        self._action_idx = 0
        self._action_progress = 0.0
        self._action_dt = 0.02 / max(action.duration, 0.02)
        self._action_timer.start(20)

    def _action_tick(self):
        """Advance one interpolation step toward the next waypoint."""
        self._action_progress += self._action_dt
        if self._action_progress >= 1.0:
            self._action_idx += 1
            if self._action_idx >= len(self._action_wps) - 1:
                # Snap to final waypoint and stop
                for s, deg in zip(self.sliders, self._action_wps[-1]):
                    s.set_value(deg)
                self._update_fk()
                self._action_timer.stop()
                return
            self._action_progress = 0.0

        wp_a = self._action_wps[self._action_idx]
        wp_b = self._action_wps[self._action_idx + 1]
        t = self._action_progress
        for s, a, b in zip(self.sliders, wp_a, wp_b):
            s.set_value(a + (b - a) * t)
        self._update_fk()

    def _send_action(self, cmd: bytes):
        """Send a one-shot serial command."""
        if self._serial and self._serial.is_open:
            try:
                self._serial.write(cmd)
            except Exception:
                pass

    # ── gamepad setup ----------------------------------------------------------

    def _init_gamepad(self, parent_layout):
        try:
            from .gamepad import GamepadManager
        except ImportError:
            return
        if not GamepadManager.available():
            return

        gp_box = QGroupBox("Gamepad")
        gp_lay = QVBoxLayout(gp_box)

        self.gp_status = QLabel("Searching ...")
        self.gp_status.setStyleSheet("color: #6c7086;")
        gp_lay.addWidget(self.gp_status)

        hint = QLabel("L-stick / R-stick: joints\nA: mode  B: reset\nX: trail  Y: clear")
        hint.setStyleSheet("color: #585b70; font-size: 11px;")
        gp_lay.addWidget(hint)

        parent_layout.addWidget(gp_box)

        self._gamepad = GamepadManager(parent=self)
        self._gamepad.connected.connect(self._gp_connected)
        self._gamepad.disconnected.connect(self._gp_disconnected)
        self._gamepad.axes_updated.connect(self._on_gp_axes)
        self._gamepad.button_pressed.connect(self._on_gp_button)

    def _gp_connected(self, name):
        self.gp_status.setText(name)
        self.gp_status.setStyleSheet("color: #a6e3a1;")

    def _gp_disconnected(self):
        self.gp_status.setText("Disconnected")
        self.gp_status.setStyleSheet("color: #6c7086;")

    # ── gamepad input ----------------------------------------------------------

    # Default mapping:  left-stick-Y → joint 0,  right-stick-Y → joint 1, …
    _FK_AXIS_MAP = [1, 3]        # pygame axis indices per joint
    _FK_SPEED    = 1.2           # degrees per tick at full deflection
    _IK_SPEED    = 0.0015        # metres per tick at full deflection

    def _on_gp_axes(self, axes: dict):
        if self.fk_btn.isChecked():
            moved = False
            for i, slider in enumerate(self.sliders):
                if i >= len(self._FK_AXIS_MAP):
                    break
                val = axes.get(self._FK_AXIS_MAP[i], 0.0)
                if val == 0.0:
                    continue
                delta = -val * self._FK_SPEED          # invert Y
                new = slider.get_deg() + delta
                new = max(slider.joint.min_deg, min(slider.joint.max_deg, new))
                slider.set_value(new)
                moved = True
            if moved:
                self._update_fk()
        else:
            # IK: left stick → EEF velocity
            vx = axes.get(0, 0.0)
            vy = -axes.get(1, 0.0)                     # invert Y
            if (vx or vy) and self._last_eef:
                tx = self._last_eef[0] + vx * self._IK_SPEED
                ty = self._last_eef[1] + vy * self._IK_SPEED
                self._on_ik_request(tx, ty)

    def _on_gp_button(self, btn: int):
        # Check model button_map first
        bmap = self.model.button_map()
        name = bmap.get(btn)
        if name:
            lookup = getattr(self, "_action_lookup", {})
            s_lookup = getattr(self, "_serial_action_lookup", {})
            if name in lookup:
                self._start_action(lookup[name])
                return
            if name in s_lookup:
                self._send_action(s_lookup[name].command)
                return

        # Default GUI bindings
        if btn == 0:                                    # A / Cross
            if self.fk_btn.isChecked():
                self.ik_btn.setChecked(True)
            else:
                self.fk_btn.setChecked(True)
        elif btn == 1:                                  # B / Circle
            self._reset()
        elif btn == 2:                                  # X / Square
            self.trail_cb.setChecked(not self.trail_cb.isChecked())
        elif btn == 3:                                  # Y / Triangle
            self.canvas.clear_trail()

    # ── slots ------------------------------------------------------------------

    def _mode_changed(self, checked):
        ik_mode = self.ik_btn.isChecked()
        self.canvas.set_ik_mode(ik_mode)
        for s in self.sliders:
            s.slider.setEnabled(not ik_mode)
        if ik_mode:
            self.canvas.setCursor(Qt.CrossCursor)
        else:
            self.canvas.setCursor(Qt.ArrowCursor)

    def _on_slider(self, _value):
        self._update_fk()

    def _on_ik_request(self, x: float, y: float):
        current = [np.deg2rad(s.get_deg()) for s in self.sliders]
        result = self.model.inverse_kinematics(x, y, current)
        if result is None:
            return
        for s, angle_rad in zip(self.sliders, result):
            s.set_value(np.rad2deg(angle_rad))
        self._update_fk()

    def _reset(self):
        for s in self.sliders:
            s.set_value(s.joint.default_deg)
        self._update_fk()

    def _update_fk(self):
        angles_rad = [np.deg2rad(s.get_deg()) for s in self.sliders]
        try:
            points = self.model.forward_kinematics(angles_rad)
        except Exception:
            return
        self.canvas.update_robot(points)
        eef = points.get(self.model.eef_name)
        if eef:
            self._last_eef = eef
            self.eef_x_label.setText(f"X: {eef[0]*1000:7.1f} mm")
            self.eef_y_label.setText(f"Y: {eef[1]*1000:7.1f} mm")

    def closeEvent(self, event):
        if hasattr(self, "_action_timer"):
            self._action_timer.stop()
        if self._serial and self._serial.is_open:
            self._serial.close()
        if self._gamepad:
            self._gamepad.cleanup()
        super().closeEvent(event)


# ── Public launcher ──────────────────────────────────────────────────────────

def launch(model: RobotModel, title: str = "Robot Viewer"):
    """Create a QApplication and open the viewer. Blocks until closed."""
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLE)
    win = RobotViewer(model, title=title)
    win.show()
    sys.exit(app.exec_())
