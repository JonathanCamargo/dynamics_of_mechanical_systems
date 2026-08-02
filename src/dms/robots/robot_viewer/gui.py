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
# Deliberately no pyplot import: this canvas is embedded in Qt, and pyplot
# would start a second, competing figure-manager for figures we own.
from matplotlib.backends.backend_qt5agg import (      # noqa: E402
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.figure import Figure                   # noqa: E402

from PyQt5.QtWidgets import (                          # noqa: E402
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QCheckBox, QButtonGroup,
    QSplitter, QGroupBox, QSizePolicy, QComboBox, QFileDialog,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal         # noqa: E402
from PyQt5.QtGui import QFont                          # noqa: E402

from ..model import RobotModel                           # noqa: E402
from .serial_link import SerialLink                       # noqa: E402

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
        self._home_lims = (tuple(xlim), tuple(ylim))
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
        self._pan = None          # (px, py, xlim, ylim) captured at press

        # Latest un-solved target; drained by _flush_target at IK_RATE_MS.
        # Only runs while dragging, so it costs nothing at rest.
        self._pending_target = None
        self._ik_timer = QTimer(self)
        self._ik_timer.setInterval(self.IK_RATE_MS)
        self._ik_timer.timeout.connect(self._flush_target)
        self.mpl_connect("button_press_event", self._on_press)
        self.mpl_connect("motion_notify_event", self._on_motion)
        self.mpl_connect("button_release_event", self._on_release)
        self.mpl_connect("scroll_event", self._on_scroll)

    # -- view (zoom / pan) ------------------------------------------------------

    ZOOM_STEP = 1.15        # per wheel notch

    def _set_view(self, xlim, ylim):
        """Apply new limits and force a full draw.

        The blitted background caches the axes, grid and ticks, so it is
        stale the moment the limits move -- draw_idle() re-runs the full
        draw, and _on_draw() recaptures the background from it.
        """
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.draw_idle()

    def reset_view(self):
        self._set_view(*self._home_lims)

    def _on_scroll(self, event):
        """Wheel zoom, anchored at the cursor so it stays over the same point."""
        if event.inaxes != self.ax:
            return
        f = 1 / self.ZOOM_STEP if event.button == "up" else self.ZOOM_STEP
        cx, cy = event.xdata, event.ydata
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        # Scale x and y by the same factor so aspect="equal" keeps the box shape.
        self._set_view(
            (cx + (x0 - cx) * f, cx + (x1 - cx) * f),
            (cy + (y0 - cy) * f, cy + (y1 - cy) * f),
        )

    # -- blitting ---------------------------------------------------------------

    def _on_draw(self, event):
        """Called after every full figure draw (initial render + resize).
        Capture the static background, then redraw animated artists so
        they stay visible after a resize / maximize."""
        self._bg = self.copy_from_bbox(self.ax.bbox)
        for artist in self._animated_artists:
            self.ax.draw_artist(artist)
        # No blit() here.  We are already inside a paint, and asking for
        # another one re-enters it ("QWidget::repaint: Recursive repaint
        # detected").  The in-flight draw flushes these artists for us.

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
            self._stop_target_drag()
            self._target.set_data([], [])
            self._blit()

    # -- target coalescing ------------------------------------------------------
    #
    # A mouse reports at 125 Hz, up to 1000 Hz for a gaming mouse, while one
    # solve-and-redraw costs ~5 ms windowed and ~15 ms maximized.  Solving per
    # event queues them and the arm falls further and further behind the
    # cursor.  So motion only *records* the latest target and the timer below
    # solves whatever is newest, dropping the rest: cost stops depending on
    # mouse rate or canvas size.

    IK_RATE_MS = 16          # ~60 Hz

    def _queue_target(self, x, y):
        self._pending_target = (x, y)

    def _flush_target(self):
        if self._pending_target is None:
            return
        x, y = self._pending_target
        self._pending_target = None
        self._target.set_data([x], [y])
        self.ik_requested.emit(x, y)

    def _stop_target_drag(self):
        self._dragging = False
        self._ik_timer.stop()
        self._pending_target = None

    def _on_press(self, event):
        if event.inaxes != self.ax:
            return
        if event.button == 3 and event.dblclick:
            self.reset_view()
            return
        # Left picks a cartesian target, right pans -- never both on one
        # button, so neither gesture has to guess at the current mode.
        if event.button == 1 and self._ik_active:
            self._dragging = True
            self._queue_target(event.xdata, event.ydata)
            self._flush_target()        # a click acts immediately
            self._ik_timer.start()
        elif event.button == 3:
            self._pan = (
                event.x, event.y,
                self.ax.get_xlim(), self.ax.get_ylim(),
            )

    def _on_motion(self, event):
        if self._dragging and event.inaxes == self.ax:
            self._queue_target(event.xdata, event.ydata)
        elif self._pan is not None:
            px, py, (x0, x1), (y0, y1) = self._pan
            # Convert the pixel delta with the limits captured at press, so
            # the maths stays valid while the limits move underneath us.
            bb = self.ax.bbox
            dx = -(event.x - px) * (x1 - x0) / bb.width
            dy = -(event.y - py) * (y1 - y0) / bb.height
            self._set_view((x0 + dx, x1 + dx), (y0 + dy, y1 + dy))

    def _on_release(self, event):
        if self._dragging:
            self._flush_target()        # land exactly where the drag ended
            self._stop_target_drag()
        self._pan = None

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
        # Flipped-U ("n"): a full-width bar on top, then three columns with
        # the live plot in the middle, framed by the two control panels.
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        self.canvas = RobotCanvas(model)
        self.canvas.setMinimumWidth(360)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.ik_requested.connect(self._on_ik_request)

        # ── Top bar: actions + live end-effector readout ──
        top_bar = QHBoxLayout()
        top_bar.setSpacing(6)
        self._init_actions(top_bar, horizontal=True)
        top_bar.addStretch()

        mono = QFont("Consolas", 12)
        eef_title = QLabel("End effector")
        eef_title.setStyleSheet("color: #6c7086;")
        top_bar.addWidget(eef_title)
        self.eef_x_label = QLabel("X:  --- mm")
        self.eef_y_label = QLabel("Y:  --- mm")
        for lb in (self.eef_x_label, self.eef_y_label):
            lb.setFont(mono)
            top_bar.addWidget(lb)
        root.addLayout(top_bar)

        splitter = QSplitter(Qt.Horizontal)
        self._splitter = splitter
        root.addWidget(splitter, stretch=1)

        # ── Left column: data in and out ──────────────────
        io_panel = QWidget()
        io_panel.setMinimumWidth(190)
        # Expanding, like the canvas -- otherwise the canvas absorbs every
        # extra pixel and the splitter stretch factors never come into play.
        io_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        io_layout = QVBoxLayout(io_panel)
        io_layout.setContentsMargins(4, 4, 4, 4)
        splitter.addWidget(io_panel)

        # Equal stretch between the groups rather than one dead gap at the
        # bottom, so a tall window spreads them instead of bunching them.
        for init in (self._init_serial, self._init_trajectory,
                     self._init_gamepad):
            init(io_layout)
            io_layout.addStretch(1)

        # ── Centre: the live plot ─────────────────────────
        splitter.addWidget(self.canvas)

        # ── Right column: live control ────────────────────
        ctrl = QWidget()
        ctrl.setMinimumWidth(210)
        ctrl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
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
            joints_lay.addStretch(1)
        # The one box worth growing: extra height becomes slider spacing.
        ctrl_layout.addWidget(joints_box, stretch=1)

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

        # Starting proportions; resizeEvent() keeps whatever ratio is current.
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 5)
        splitter.setStretchFactor(2, 1)
        QTimer.singleShot(0, lambda: self._split_ratio(0.16, 0.68, 0.16))

        # Initial draw -- deferred so the canvas has a valid pixel buffer
        QTimer.singleShot(0, self._update_fk)

    # ── serial setup -----------------------------------------------------------

    def _init_serial(self, parent_layout):
        cfg = self.model.serial_config()
        if not cfg or not SerialLink.available():
            return

        self._serial = SerialLink(self)
        self._serial.status.connect(self._serial_show_status)

        box = QGroupBox("Serial")
        lay = QVBoxLayout(box)

        row = QHBoxLayout()
        self._port_combo = QComboBox()
        self._port_combo.setMinimumWidth(100)
        # 'activated' fires only on user interaction, unlike currentIndexChanged
        # which also fires while _serial_refresh repopulates the list.
        self._port_combo.activated.connect(self._on_port_chosen)
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
        self._serial_status.setStyleSheet(f"color: {SerialLink.IDLE_COLOUR};")
        lay.addWidget(self._serial_status)

        parent_layout.addWidget(box)
        self._serial_refresh()

        # Send timer.  Hobby servos do not resolve anything like 50 Hz of
        # distinct targets, and the slower rate gives the device far more
        # room to drain between writes.
        self._servo_timer = QTimer(self)
        self._servo_timer.timeout.connect(self._serial_tick)
        self._servo_timer.start(cfg.get("send_interval_ms", 40))  # 25 Hz

    def _serial_show_status(self, message: str, colour: str):
        self._serial_status.setText(message)
        self._serial_status.setStyleSheet(f"color: {colour};")

    def _on_port_chosen(self, _index):
        """Remember an explicit user choice so rescans cannot override it."""
        self._chosen_port = self._port_combo.currentData()

    def _serial_refresh(self):
        if not hasattr(self, "_port_combo"):
            return
        self._port_combo.clear()
        for label, device in SerialLink.list_ports():
            self._port_combo.addItem(label, device)

        # Restore the user's port across rescans and reconnects.  Only fall
        # back to the auto-pick (first real USB device) if they never chose
        # one, or if the one they chose is genuinely gone from the bus.
        chosen = getattr(self, "_chosen_port", None)
        if chosen is not None:
            idx = self._port_combo.findData(chosen)
            if idx >= 0:
                self._port_combo.setCurrentIndex(idx)

    def _serial_toggle(self):
        if self._serial.is_open:
            self._serial.close()
            self._serial_btn.setText("Connect")
            self._port_combo.setEnabled(True)
            return

        port = self._port_combo.currentData()
        if not port:
            return
        if self._serial.open(port, self.model.serial_config()):
            # Connecting is a commitment to this port: stick to it for every
            # later rescan and reconnect, auto-picked or not.
            self._chosen_port = port
            self._serial_btn.setText("Disconnect")
            self._port_combo.setEnabled(False)

    def _serial_tick(self):
        """Send the current target to the servos; called on the send timer."""
        target = [s.get_deg() for s in self.sliders]
        self._serial.send(self.model.serial_command(target))

    # ── actions setup ----------------------------------------------------------

    def _init_actions(self, parent_layout, horizontal=False):
        """Build the action buttons.

        ``horizontal=True`` adds them straight into *parent_layout* as a bare
        row (for the top bar); otherwise they go in a titled group box.
        """
        acts = self.model.actions()
        s_acts = self.model.serial_actions()

        if acts or s_acts:
            if horizontal:
                lay = parent_layout
            else:
                box = QGroupBox("Actions")
                lay = QVBoxLayout(box)

            for act in acts:
                btn = QPushButton(act.name)
                btn.clicked.connect(lambda _, a=act: self._start_action(a))
                lay.addWidget(btn)

            for sa in s_acts:
                btn = QPushButton(sa.name)
                btn.clicked.connect(
                    lambda _, cmd=sa.command: self._send_action(cmd)
                )
                lay.addWidget(btn)

            if not horizontal:
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
        if self._serial:
            self._serial.send_now(cmd)

    # ── trajectory playback ----------------------------------------------------

    def _init_trajectory(self, parent_layout):
        from .trajectory import TrajectoryPlayer

        box = QGroupBox("Trajectory")
        lay = QVBoxLayout(box)

        self._traj_load_btn = QPushButton("Load CSV ...")
        self._traj_load_btn.clicked.connect(self._traj_load)
        lay.addWidget(self._traj_load_btn)

        btn_row = QHBoxLayout()
        self._traj_play_btn = QPushButton("Play")
        self._traj_play_btn.clicked.connect(self._traj_play)
        self._traj_play_btn.setEnabled(False)
        self._traj_stop_btn = QPushButton("Stop")
        self._traj_stop_btn.clicked.connect(self._traj_stop)
        self._traj_stop_btn.setEnabled(False)
        btn_row.addWidget(self._traj_play_btn)
        btn_row.addWidget(self._traj_stop_btn)
        lay.addLayout(btn_row)

        self._traj_status = QLabel("No trajectory loaded")
        self._traj_status.setStyleSheet("color: #6c7086; font-size: 11px;")
        self._traj_status.setWordWrap(True)
        lay.addWidget(self._traj_status)

        parent_layout.addWidget(box)

        self._trajectory = TrajectoryPlayer(parent=self)
        self._trajectory.sample.connect(self._traj_apply)
        self._trajectory.finished.connect(self._traj_finished)

    def _traj_load(self):
        from .trajectory import load_trajectory

        path, _ = QFileDialog.getOpenFileName(
            self, "Load trajectory CSV", "", "CSV files (*.csv);;All files (*)"
        )
        if not path:
            return
        n = len(self.model.joint_info())
        try:
            times, angles = load_trajectory(path, n)
        except (ValueError, OSError) as exc:
            self._traj_status.setText(str(exc))
            self._traj_status.setStyleSheet("color: #f38ba8; font-size: 11px;")
            self._traj_play_btn.setEnabled(False)
            return

        self._trajectory.load(times, angles)
        import os
        self._traj_status.setText(
            f"{os.path.basename(path)}  ({len(times)} pts, "
            f"{self._trajectory.duration:.1f} s)"
        )
        self._traj_status.setStyleSheet("color: #a6e3a1; font-size: 11px;")
        self._traj_play_btn.setEnabled(True)

    def _traj_play(self):
        if self._trajectory.duration <= 0:
            return
        # FK mode so slider writes drive the pose directly.
        if self.ik_btn.isChecked():
            self.fk_btn.setChecked(True)
        self._traj_play_btn.setEnabled(False)
        self._traj_stop_btn.setEnabled(True)
        self._traj_load_btn.setEnabled(False)
        self._trajectory.play()

    def _traj_stop(self):
        self._trajectory.stop()

    def _traj_apply(self, angles):
        """Write interpolated angles to the sliders (feeds serial + FK view)."""
        for s, deg in zip(self.sliders, angles):
            s.set_value(deg)
        self._update_fk()

    def _traj_finished(self):
        self._traj_play_btn.setEnabled(True)
        self._traj_stop_btn.setEnabled(False)
        self._traj_load_btn.setEnabled(True)

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

    # ── splitter proportions ---------------------------------------------------

    def _split_ratio(self, *fractions):
        w = self._splitter.width()
        if w > 0:
            self._splitter.setSizes([max(1, int(f * w)) for f in fractions])

    def resizeEvent(self, event):
        """Keep the three columns' proportions when the window is resized.

        Splitter stretch factors alone do not survive this: FigureCanvas
        reports its *current* pixel size as its sizeHint, so it ratchets up
        and swallows the extra width before the factors are consulted.
        Rescaling the current sizes also preserves any manual handle drag.
        """
        super().resizeEvent(event)
        # setSizes() resizes children, which can re-enter this handler.
        if getattr(self, "_in_resize", False):
            return
        sizes = self._splitter.sizes()
        total, new_total = sum(sizes), self._splitter.width()
        if total > 0 and new_total > 0 and abs(new_total - total) > 1:
            self._in_resize = True
            try:
                self._splitter.setSizes(
                    [max(1, int(s * new_total / total)) for s in sizes]
                )
            finally:
                self._in_resize = False

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
        if self._serial:
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
