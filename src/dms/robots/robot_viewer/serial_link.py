"""
Serial link to a servo controller.

Split out of the viewer window so the protocol behaviour -- draining, retry,
back-pressure, recovery -- can be reasoned about and tested without a GUI.

The hard-won details, all of which cost real debugging time:

* **Replies must be drained every tick, even when we send nothing.**  On a
  native-USB-CDC board the device's own ``Serial.print`` blocks when the host
  is not reading.  A blocked device stops reading our commands, so every write
  then times out -- permanently.  A resting arm sends no commands, which is
  precisely when unread replies would pile up unnoticed.

* **A write timeout is back-pressure, not an error.**  It means the device is
  consuming slower than we send.  Retry, never disconnect.

* **A wedged pipe does not clear itself.**  Nothing host-side can unstick a
  blocked sketch, so after sustained back-pressure the port is closed and
  reopened; the DTR toggle resets most boards.
"""

import sys

from PyQt5.QtCore import QObject, pyqtSignal

try:
    import serial
    import serial.tools.list_ports
    _HAS_SERIAL = True
except ImportError:
    _HAS_SERIAL = False


class SerialLink(QObject):
    """Owns the port, the send policy, and the recovery policy."""

    #: (message, colour) -- the viewer renders this into its status label.
    status = pyqtSignal(str, str)

    OK_COLOUR = "#a6e3a1"
    WARN_COLOUR = "#f9e2af"
    ERROR_COLOUR = "#f38ba8"
    IDLE_COLOUR = "#6c7086"

    #: Consecutive hard errors tolerated before dropping the connection.
    #: Write timeouts do not count -- they are handled as back-pressure.
    MAX_FAILS = 5

    #: Consecutive busy ticks before bouncing the port (25 Hz -> ~1 s).
    BOUNCE_AFTER = 25

    def __init__(self, parent=None):
        super().__init__(parent)
        self._port = None
        self._cfg = {}
        self._device = None
        self._last_cmd = None
        self._fails = 0
        self._busy = 0

    # ── availability ──────────────────────────────────────────────────

    @staticmethod
    def available() -> bool:
        return _HAS_SERIAL

    @staticmethod
    def list_ports() -> list[tuple[str, str]]:
        """Return ``(label, device)`` pairs, real USB hardware first.

        Real USB devices have a VID/PID; Bluetooth SPP ports do not.  Those
        exist whether or not anything is paired behind them, and writing to an
        unpaired one never completes -- every write simply times out.  So they
        sort last and are labelled, to keep them from being picked by default.
        """
        if not _HAS_SERIAL:
            return []
        from serial.tools.list_ports import comports
        out = []
        for p in sorted(comports(), key=lambda p: (p.vid is None, p.device)):
            label = f"{p.device} - {p.description}"
            if p.vid is None:
                label += "  (no device)"
            out.append((label, p.device))
        return out

    @staticmethod
    def port_present(device: str) -> bool:
        """Whether *device* is still enumerated on the bus."""
        return any(dev == device for _, dev in SerialLink.list_ports())

    # ── connection ────────────────────────────────────────────────────

    @property
    def is_open(self) -> bool:
        return bool(self._port and self._port.is_open)

    def open(self, device: str, cfg: dict) -> bool:
        """Open *device*. Returns True on success, emitting status either way."""
        if not _HAS_SERIAL or not device:
            return False
        self._device, self._cfg = device, cfg
        try:
            self._port = serial.Serial(
                device,
                baudrate=cfg.get("baudrate", 115200),
                write_timeout=cfg.get("write_timeout", 0.2),
            )
        except Exception as exc:
            self._port = None
            self.status.emit(str(exc)[:60], self.ERROR_COLOUR)
            return False
        self._reset_counters()
        self.status.emit(device, self.OK_COLOUR)
        return True

    def close(self):
        if self._port:
            try:
                self._port.close()
            except Exception:
                pass
        self._port = None
        self._reset_counters()
        self.status.emit("Disconnected", self.IDLE_COLOUR)

    def _reset_counters(self):
        self._last_cmd = None
        self._fails = 0
        self._busy = 0

    # ── the send path ─────────────────────────────────────────────────

    def send(self, cmd: bytes | None):
        """Drain replies, then write *cmd* if it differs from the last one.

        Safe to call on a timer regardless of connection state.
        """
        if not self.is_open:
            return

        # Before the early returns below: see the module docstring on why a
        # resting arm still has to consume whatever the device sent.
        try:
            waiting = self._port.in_waiting
            if waiting:
                self._port.read(waiting)
        except Exception:
            pass

        if cmd is None or cmd == self._last_cmd:
            return

        try:
            self._port.write(cmd)
        except serial.SerialTimeoutException:
            self._on_busy()
            return
        except serial.SerialException as exc:
            self._on_error(str(exc))
            return

        # Only now is the command really sent.  Both counters are consecutive
        # counts, so a success clears them.
        self._last_cmd = cmd
        if self._fails or self._busy:
            self._reset_counters()
            self._last_cmd = cmd
            self.status.emit(self._device, self.OK_COLOUR)

    def send_now(self, cmd: bytes):
        """Send a one-shot command, bypassing the repeat suppression.

        Pressing "Close Gripper" twice must send twice, so these must not go
        through send()'s identical-command check.
        """
        if not self.is_open or not cmd:
            return
        try:
            self._port.write(cmd)
        except serial.SerialTimeoutException:
            self._on_busy()
        except serial.SerialException as exc:
            self._on_error(str(exc))

    def _on_busy(self):
        """Device is not draining. Keep the command queued and retry."""
        self._busy += 1
        self.status.emit(f"device busy (x{self._busy})", self.WARN_COLOUR)
        if self._busy % self.BOUNCE_AFTER == 0:
            self.bounce()

    def _on_error(self, message: str):
        """Hard error: report, and give up only if they keep coming."""
        if self._device and not self.port_present(self._device):
            message = f"{self._device} vanished from USB -- check power"
        print(f"[serial] {message}", file=sys.stderr)

        self._fails += 1
        if self._fails >= self.MAX_FAILS:
            self.close()
            self.status.emit(f"Dropped: {message[:50]}", self.ERROR_COLOUR)
        else:
            self.status.emit(
                f"{self._fails}/{self.MAX_FAILS}: {message[:60]}",
                self.WARN_COLOUR,
            )

    def bounce(self):
        """Close and reopen to clear a wedged pipe.

        Once the device stops draining, nothing on the host side recovers it.
        Reopening drops and re-raises DTR, which resets most boards and
        restarts the sketch with empty buffers.
        """
        if not self.is_open:
            return
        device, cfg = self._device, self._cfg
        try:
            self._port.close()
        except Exception:
            pass
        self._port = None
        print(f"[serial] bouncing {device} to clear a wedged pipe",
              file=sys.stderr)
        self.open(device, cfg)
