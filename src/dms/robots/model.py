"""
Abstract robot model interface.

Implement ``RobotModel`` to plug any 2-D planar mechanism into the viewer.

Two ways to create a robot
--------------------------
1. **Pure numeric** -- subclass ``RobotModel`` directly and implement
   ``forward_kinematics()`` with plain numpy (or anything else).

2. **Symbolic (sympy)** -- subclass ``SymbolicRobotModel`` (from
   ``dms.robots.symbolic``).  Define your mechanism symbolically,
   call ``_lambdify_points()``, and FK is provided for you.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class JointInfo:
    """Description of one controllable (actuated) joint."""
    name: str
    min_deg: float
    max_deg: float
    default_deg: float


@dataclass
class LinkDrawing:
    """How to draw a single link: a line between two named points."""
    start: str
    end: str
    color: str = "#cccccc"
    linewidth: float = 2.5


@dataclass
class PointDrawing:
    """How to draw a named point / joint marker."""
    name: str
    marker: str = "o"
    color: str = "#ffffff"
    size: float = 6.0


@dataclass
class Action:
    """A named motion sequence through joint-angle waypoints (degrees)."""
    name: str
    waypoints: list[list[float]]
    duration: float = 1.0          # seconds per waypoint transition


@dataclass
class SerialAction:
    """A one-shot serial command (gripper, LED, etc.)."""
    name: str
    command: bytes


@dataclass
class ServoCal:
    """Affine map from a joint angle to a servo command, both in degrees.

        servo = servo_ref + gain * (q_deg - q_ref)

    Write down a pose you actually measured rather than a bare offset:
    "joint A vertical (q=90) reads 110 on the servo" becomes
    ``ServoCal(8, q_ref=90, servo_ref=110)``.

    ``gain`` carries the sign -- use -1 when the servo turns against the
    joint's positive sense -- and the scale, for a geared or lever-driven
    joint where one degree of joint is not one degree of servo.

    This is per physical arm: it changes every time a horn comes off a
    spline, unlike the kinematics.
    """
    id: int
    q_ref: float = 0.0
    servo_ref: float = 90.0
    gain: float = 1.0
    lo: float = 0.0
    hi: float = 180.0

    def raw(self, q_deg: float) -> float:
        """Servo command, unclamped -- what the joint angle *asks* for."""
        return self.servo_ref + self.gain * (q_deg - self.q_ref)

    def to_servo(self, q_deg: float) -> float:
        """Servo command, clamped into ``[lo, hi]``."""
        return min(self.hi, max(self.lo, self.raw(q_deg)))

    def to_joint(self, servo_deg: float) -> float:
        """Inverse map -- for checking a calibration against the real arm."""
        return self.q_ref + (servo_deg - self.servo_ref) / self.gain

    def servo_range_for(self, joint: "JointInfo") -> tuple[float, float]:
        """Servo interval swept by a joint's full travel, low first.

        Sorted because a negative ``gain`` maps ``min_deg`` to the upper end.
        """
        a, b = self.raw(joint.min_deg), self.raw(joint.max_deg)
        return (min(a, b), max(a, b))

    @classmethod
    def from_two_points(cls, id, q1, s1, q2, s2, **kw) -> "ServoCal":
        """Derive gain and offset from two jogged-and-measured poses.

        Recovers the sign from the measurements, so you never have to reason
        about which way the horn turns.
        """
        return cls(id, q_ref=q1, servo_ref=s1, gain=(s2 - s1) / (q2 - q1), **kw)


class RobotModel(ABC):
    """
    Pure interface for any 2-D planar robot or mechanism.

    Implement every ``@abstractmethod`` below, then call
    ``dms.robots.robot_viewer.launch(YourRobot())`` to open the GUI.

    Abstract methods (must implement)
    ----------------------------------
    - ``joint_info()``           -- actuated DOFs with limits
    - ``forward_kinematics()``   -- joint angles → named point positions
    - ``get_links()``            -- visual line segments between points
    - ``get_points()``           -- visual markers at named points
    - ``view_limits()``          -- default axis limits

    Point-name contract
    --------------------
    The string names used in ``get_links()`` and ``get_points()`` **must**
    appear as keys in the dict returned by ``forward_kinematics()``.
    Call ``validate()`` after construction to verify this.

    Optional overrides
    ------------------
    - ``inverse_kinematics()``   -- default uses numerical least-squares
    - ``actions()``              -- waypoint motion sequences
    - ``serial_actions()``       -- one-shot serial commands
    - ``serial_config()``        -- enable serial port
    - ``serial_command()``       -- build bytes to send each tick
    - ``button_map()``           -- map gamepad buttons to actions
    """

    eef_name: str = "EEF"
    ik_tolerance: float = 1e-3

    #: One ``ServoCal`` per actuated joint, in ``joint_info()`` order.
    #: Set it (class attribute or in ``__init__``) and the default
    #: ``serial_command()`` below builds the packet for you.
    servo_cal: list = []

    #: Servo id of a gripper, if there is one.  Set it and the default
    #: ``serial_actions()`` and ``button_map()`` provide open/close.
    gripper_id: int | None = None

    #: Set to enable the serial link; see ``serial_config()``.
    serial_baudrate: int | None = None

    @abstractmethod
    def joint_info(self) -> list[JointInfo]:
        """Return descriptions of each actuated joint."""
        ...

    @abstractmethod
    def forward_kinematics(
        self, joint_angles_rad: list[float]
    ) -> dict[str, tuple[float, float]]:
        """Compute every named point position from the actuated angles.

        Returns a dict mapping point name → (x, y).  Every name referenced
        by ``get_links()``, ``get_points()``, and ``eef_name`` **must**
        appear as a key.
        """
        ...

    def inverse_kinematics(
        self, x: float, y: float, current_angles_rad: list[float]
    ) -> Optional[list[float]]:
        """Return actuated angles that place the EEF at (x, y), or *None*."""
        import numpy as np
        from scipy.optimize import least_squares

        joints = self.joint_info()
        x0 = np.array(current_angles_rad[:len(joints)])
        eef = self.eef_name

        def residual(thetas):
            pts = self.forward_kinematics(list(thetas))
            ex, ey = pts[eef]
            return [ex - x, ey - y]

        lb = [np.deg2rad(j.min_deg) for j in joints]
        ub = [np.deg2rad(j.max_deg) for j in joints]
        sln = least_squares(residual, x0, bounds=(lb, ub))
        if np.sqrt(sln.cost) > self.ik_tolerance:
            return None
        return list(sln.x)

    @abstractmethod
    def get_links(self) -> list[LinkDrawing]:
        """Visual specs for every link to draw."""
        ...

    @abstractmethod
    def get_points(self) -> list[PointDrawing]:
        """Visual specs for every point / joint marker to draw."""
        ...

    @abstractmethod
    def view_limits(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return ``((xmin, xmax), (ymin, ymax))`` for the default view."""
        ...

    # ── validation ────────────────────────────────────────────────────

    def validate(self):
        """Check that FK output, links, and points use consistent names.

        Call after construction to catch typos early.  Raises ``ValueError``
        with a clear message if any drawing name is missing from FK output.
        """
        import numpy as np
        angles = [np.deg2rad(j.default_deg) for j in self.joint_info()]
        pts = self.forward_kinematics(angles)

        missing = []
        for lk in self.get_links():
            if lk.start not in pts:
                missing.append(f"LinkDrawing start '{lk.start}'")
            if lk.end not in pts:
                missing.append(f"LinkDrawing end '{lk.end}'")
        for pt in self.get_points():
            if pt.name not in pts:
                missing.append(f"PointDrawing '{pt.name}'")
        if self.eef_name not in pts:
            missing.append(f"eef_name '{self.eef_name}'")

        if missing:
            raise ValueError(
                "Point names not found in forward_kinematics() output:\n  "
                + "\n  ".join(missing)
            )

        self.validate_servo_cal()

    def validate_servo_cal(self):
        """Check each joint's travel fits inside its servo's range.

        A wrong ``gain`` sign usually swings the mapped range off the end of
        ``[lo, hi]``, so this catches it at construction rather than by the
        arm driving into a hard stop.
        """
        joints = self.joint_info()
        # Joint sanity holds with or without servos attached.
        for ji in joints:
            if ji.min_deg >= ji.max_deg:
                raise ValueError(
                    f"{ji.name}: min_deg ({ji.min_deg}) must be < max_deg "
                    f"({ji.max_deg}); IK bounds require it too."
                )
            if not ji.min_deg <= ji.default_deg <= ji.max_deg:
                raise ValueError(
                    f"{ji.name}: default_deg ({ji.default_deg}) is outside "
                    f"[{ji.min_deg}, {ji.max_deg}]; the slider would clamp it."
                )

        if not self.servo_cal:
            return
        if len(self.servo_cal) != len(joints):
            raise ValueError(
                f"servo_cal has {len(self.servo_cal)} entries but there are "
                f"{len(joints)} actuated joints."
            )
        for ji, c in zip(joints, self.servo_cal):
            s_lo, s_hi = c.servo_range_for(ji)
            if s_lo < c.lo - 1e-9 or s_hi > c.hi + 1e-9:
                raise ValueError(
                    f"{ji.name}: travel [{ji.min_deg}, {ji.max_deg}] deg maps "
                    f"to servo {c.id} range [{s_lo:.1f}, {s_hi:.1f}], outside "
                    f"[{c.lo}, {c.hi}]. Wrong gain sign, or narrow the limits."
                )

    # ── convenience ───────────────────────────────────────────────────

    def plot(self, joint_angles_rad, ax=None):
        """Quick matplotlib plot of the robot at the given joint angles."""
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.gca()
        pts = self.forward_kinematics(joint_angles_rad)
        for lk in self.get_links():
            xs = [pts[lk.start][0], pts[lk.end][0]]
            ys = [pts[lk.start][1], pts[lk.end][1]]
            ax.plot(xs, ys, color=lk.color, linewidth=lk.linewidth)
        for pt in self.get_points():
            x, y = pts[pt.name]
            ax.plot(x, y, marker=pt.marker, color=pt.color, markersize=pt.size)
        ax.set_aspect("equal")

    # ── optional actions (override to enable) ───────────────────────

    def actions(self) -> list[Action]:
        """Named motion sequences. Override to add action buttons."""
        return []

    def serial_actions(self) -> list[SerialAction]:
        """One-shot serial commands (gripper, LED, etc.).

        Set ``gripper_id`` to get open/close buttons for free; override for
        anything else.
        """
        if self.gripper_id is None:
            return []
        return [
            SerialAction("Open Gripper",
                         f"Y {self.gripper_id} 100\r\n".encode()),
            SerialAction("Close Gripper",
                         f"Y {self.gripper_id} 40\r\n".encode()),
        ]

    def button_map(self) -> dict[int, str]:
        """Map gamepad button index → action/serial-action name.

        Defaults to the gripper on the shoulder buttons when there is one.
        Override to customize.
        """
        if self.gripper_id is None:
            return {}
        return {
            4: "Open Gripper",      # LB
            5: "Close Gripper",     # RB
        }

    # ── optional serial support (override to enable) ──────────────────

    def serial_config(self) -> dict:
        """Return ``{'baudrate': 9600, ...}`` to enable serial. Empty = off.

        Set ``serial_baudrate`` for the common case; override for extra
        settings such as ``write_timeout`` or ``send_interval_ms``.
        """
        if self.serial_baudrate is None:
            return {}
        return {"baudrate": self.serial_baudrate}

    def serial_command(self, joint_angles_deg: list[float]) -> bytes | None:
        """Build the bytes to send over serial, or *None* to skip.

        Default: map each joint angle through ``servo_cal`` and emit one
        ``Y <id> <deg>`` line per joint.  Override for a different protocol.
        """
        if not self.servo_cal:
            return None
        return "".join(
            f"Y {c.id} {int(round(c.to_servo(q)))}\r\n"
            for c, q in zip(self.servo_cal, joint_angles_deg)
        ).encode()
