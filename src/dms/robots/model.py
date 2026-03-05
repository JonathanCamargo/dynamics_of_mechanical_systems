"""
Abstract robot model interface.

Implement ``RobotModel`` to plug any 2-D planar mechanism into the viewer.
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


class RobotModel(ABC):
    """
    Implement this for any 2-D planar robot or mechanism.

    Minimal example
    ---------------
    1. Define joints (actuated DOFs) in ``joint_info()``.
    2. Implement ``forward_kinematics()`` so that given the actuated
       joint angles it returns *all* named point positions.
    3. Implement ``inverse_kinematics()`` to move the end-effector.
    4. Describe the visual representation via ``get_links()`` /
       ``get_points()``.

    Then call ``dms.robots.robot_viewer.launch(YourRobot())`` to open the GUI.
    """

    eef_name: str = "EEF"
    ik_tolerance: float = 1e-3

    @abstractmethod
    def joint_info(self) -> list[JointInfo]:
        """Return descriptions of each actuated joint."""
        ...

    def forward_kinematics(
        self, joint_angles_rad: list[float]
    ) -> dict[str, tuple[float, float]]:
        """Compute every named point position from the actuated angles."""
        return {
            name: tuple(fn(*joint_angles_rad))
            for name, fn in self._pts_fn.items()
        }

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

    def end_effector_name(self) -> str:
        """Name of the end-effector point (must appear in FK output)."""
        return self.eef_name

    @abstractmethod
    def view_limits(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return ``((xmin, xmax), (ymin, ymax))`` for the default view."""
        ...

    # ── shared helpers ────────────────────────────────────────────────

    def _lambdify_points(self, theta, pts, N, params):
        """Lambdify a dict of sympy vectors into fast numpy callables."""
        import sympy
        self._pts_fn = {}
        for name, vec in pts.items():
            x_expr = vec.dot(N.x).subs(params)
            y_expr = vec.dot(N.y).subs(params)
            self._pts_fn[name] = sympy.lambdify(
                theta, [x_expr, y_expr], "numpy"
            )

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

    # ── optional serial support (override to enable) ──────────────────

    def serial_config(self) -> dict:
        """Return ``{'baudrate': 9600, ...}`` to enable serial. Empty = off."""
        return {}

    def serial_command(self, joint_angles_deg: list[float]) -> bytes | None:
        """Build the bytes to send over serial, or *None* to skip."""
        return None
