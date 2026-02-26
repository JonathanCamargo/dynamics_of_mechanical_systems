"""
Optional gamepad/joystick support via pygame.

If pygame is not installed the GUI still works -- gamepad features are
simply disabled.
"""

import os

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

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._timer.start(poll_ms)

        self._try_connect()

    # ------------------------------------------------------------------

    def _try_connect(self):
        pygame.joystick.quit()
        pygame.joystick.init()
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
