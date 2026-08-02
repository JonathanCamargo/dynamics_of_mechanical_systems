"""
Optional gamepad/joystick support via pygame.

If pygame is not installed the GUI still works -- gamepad features are
simply disabled.
"""

import os
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from PyQt5.QtCore import QObject, QTimer, pyqtSignal

try:
    import pygame
    _HAS_PYGAME = True
except ImportError:
    _HAS_PYGAME = False


class GamepadManager(QObject):
    """Polls the first connected gamepad at ~60 Hz and emits Qt signals."""

    # {axis_index: float} for every axis (0.0 inside deadzone)
    axes_updated = pyqtSignal(dict)
    # Rising-edge only -- emitted once per press
    button_pressed = pyqtSignal(int)
    # Connection bookkeeping
    connected = pyqtSignal(str)       # controller name
    disconnected = pyqtSignal()

    #: Seconds between device scans while nothing is connected.
    RESCAN_S = 2.0

    def __init__(self, poll_ms: int = 16, deadzone: float = 0.12, parent=None):
        super().__init__(parent)
        if not _HAS_PYGAME:
            return

        pygame.joystick.init()
        if not pygame.get_init():
            pygame.init()

        self._joy = None
        self._deadzone = deadzone
        self._prev_buttons: set[int] = set()
        self._last_scan = time.monotonic()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._timer.start(poll_ms)

        self._try_connect()

    # ------------------------------------------------------------------

    def _try_connect(self):
        """Grab the first joystick, if one is attached.

        Deliberately no ``pygame.joystick.quit()/init()`` cycle here.  That
        used to be the way to re-detect hot-plugged devices, but SDL2 reports
        hotplug through the event queue that ``_poll`` already pumps, so
        ``get_count()`` is current without it.  The cycle also had two real
        costs: it re-enumerated every device (measured at ~540 ms on a machine
        with a SpaceMouse attached), and it invalidated the Joystick objects
        held by *any* other GamepadManager in the process -- polling one of
        those freed handles segfaults the interpreter.
        """
        if pygame.joystick.get_count() > 0:
            self._joy = pygame.joystick.Joystick(0)
            self._joy.init()
            self.connected.emit(self._joy.get_name())
            self._prev_buttons = set()

    def _poll(self):
        if not _HAS_PYGAME:
            return

        try:
            pygame.event.pump()
        except Exception:
            return

        if self._joy is None:
            # pygame.joystick.quit()/init() re-enumerates every device and can
            # take hundreds of milliseconds.  At the 16 ms poll rate that would
            # freeze the GUI thread outright, so scan only occasionally.
            now = time.monotonic()
            if now - self._last_scan >= self.RESCAN_S:
                self._last_scan = now
                self._try_connect()
            return

        # Check still alive
        try:
            self._joy.get_name()
        except Exception:
            self._joy = None
            self.disconnected.emit()
            return

        # Axes
        axes: dict[int, float] = {}
        for i in range(self._joy.get_numaxes()):
            v = self._joy.get_axis(i)
            axes[i] = v if abs(v) > self._deadzone else 0.0
        self.axes_updated.emit(axes)

        # Buttons (rising edge)
        current: set[int] = set()
        for i in range(self._joy.get_numbuttons()):
            if self._joy.get_button(i):
                current.add(i)
        for b in current - self._prev_buttons:
            self.button_pressed.emit(b)
        self._prev_buttons = current

    # ------------------------------------------------------------------

    def cleanup(self):
        if _HAS_PYGAME:
            self._timer.stop()
            pygame.joystick.quit()

    @staticmethod
    def available() -> bool:
        return _HAS_PYGAME
