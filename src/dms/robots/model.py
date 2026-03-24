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
        """One-shot serial commands (gripper, LED, etc.). Override to add."""
        return []

    def button_map(self) -> dict[int, str]:
        """Map gamepad button index → action/serial-action name. Override to customize."""
        return {}

    # ── optional serial support (override to enable) ──────────────────

    def serial_config(self) -> dict:
        """Return ``{'baudrate': 9600, ...}`` to enable serial. Empty = off."""
        return {}

    def serial_command(self, joint_angles_deg: list[float]) -> bytes | None:
        """Build the bytes to send over serial, or *None* to skip."""
        return None
