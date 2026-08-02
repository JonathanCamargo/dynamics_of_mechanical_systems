"""
Pre-recorded trajectory playback.

Loads a CSV of time-stamped joint angles and replays it through the viewer
(or, headless, straight to serial).  Model-agnostic: it depends only on the
number of actuated joints, so the same player drives any ``RobotModel``.

CSV format
----------
One column of absolute time in **seconds**, then one column per actuated
joint in **degrees**, in ``joint_info()`` order::

    time, joint_0, joint_1
    0.0,  90,      30
    0.5,  90,     -40
    1.2,  60,     -40

A header row is optional (auto-detected).  Times must be non-decreasing;
angles between samples are linearly interpolated against wall-clock time.
"""

import csv
import numpy as np

from PyQt5.QtCore import QObject, QTimer, pyqtSignal


def load_trajectory(path: str, n_joints: int):
    """Parse a trajectory CSV into ``(times, angles_deg)`` arrays.

    Parameters
    ----------
    path : str
        CSV file with columns ``time, joint_0, ..., joint_{n-1}``.
    n_joints : int
        Expected number of joint columns (from ``len(model.joint_info())``).

    Returns
    -------
    times : (N,) float array -- absolute seconds, starting at 0.
    angles_deg : (N, n_joints) float array -- degrees.

    Raises
    ------
    ValueError
        On wrong column count, empty file, non-numeric data, or times that
        decrease.
    """
    rows = []
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        for lineno, raw in enumerate(reader, start=1):
            cells = [c.strip() for c in raw if c.strip() != ""]
            if not cells:
                continue  # skip blank lines
            try:
                rows.append([float(c) for c in cells])
            except ValueError:
                if lineno == 1:
                    continue  # header row -- skip it
                raise ValueError(
                    f"{path}: non-numeric data on line {lineno}: {raw!r}"
                )

    if not rows:
        raise ValueError(f"{path}: no data rows found.")

    expected = 1 + n_joints
    for i, r in enumerate(rows):
        if len(r) != expected:
            raise ValueError(
                f"{path}: row {i} has {len(r)} columns, expected {expected} "
                f"(1 time + {n_joints} joints)."
            )

    data = np.asarray(rows, dtype=float)
    times = data[:, 0]
    angles_deg = data[:, 1:]

    if np.any(np.diff(times) < 0):
        raise ValueError(f"{path}: time column must be non-decreasing.")

    times = times - times[0]  # normalise so playback starts at t=0
    return times, angles_deg


class TrajectoryPlayer(QObject):
    """Replays a loaded trajectory against wall-clock time.

    Emits :attr:`sample` on every tick with the interpolated joint angles
    (degrees), and :attr:`finished` once the final timestamp is reached.
    The consumer is responsible for applying the angles (e.g. writing the
    viewer's sliders), so the player stays independent of any robot.
    """

    sample = pyqtSignal(list)   # interpolated joint angles, degrees
    finished = pyqtSignal()

    def __init__(self, tick_ms: int = 20, parent=None):
        super().__init__(parent)
        self._times = None
        self._angles = None
        self._elapsed = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(tick_ms)
        self._timer.timeout.connect(self._tick)

    def load(self, times, angles):
        """Store a trajectory (see :func:`load_trajectory` for the arrays)."""
        self._times = np.asarray(times, dtype=float)
        self._angles = np.asarray(angles, dtype=float)

    @property
    def duration(self) -> float:
        return 0.0 if self._times is None else float(self._times[-1])

    @property
    def is_playing(self) -> bool:
        return self._timer.isActive()

    def play(self):
        """Start (or restart) playback from the beginning."""
        if self._times is None or len(self._times) == 0:
            return
        self._elapsed = 0.0
        # perf_counter avoids drift from accumulated tick error.
        import time
        self._t0 = time.perf_counter()
        self._timer.start()

    def stop(self):
        """Halt playback where it is."""
        if self._timer.isActive():
            self._timer.stop()
            self.finished.emit()

    def _tick(self):
        import time
        elapsed = time.perf_counter() - self._t0

        if elapsed >= self._times[-1]:
            self.sample.emit([float(a) for a in self._angles[-1]])
            self._timer.stop()
            self.finished.emit()
            return

        angles = [
            float(np.interp(elapsed, self._times, self._angles[:, j]))
            for j in range(self._angles.shape[1])
        ]
        self.sample.emit(angles)
