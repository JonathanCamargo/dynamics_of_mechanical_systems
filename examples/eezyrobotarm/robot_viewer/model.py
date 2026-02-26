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

    Then call ``robot_viewer.launch(YourRobot())`` to open the GUI.
    """

    @abstractmethod
    def joint_info(self) -> list[JointInfo]:
        """Return descriptions of each actuated joint."""
        ...

    @abstractmethod
    def forward_kinematics(
        self, joint_angles_rad: list[float]
    ) -> dict[str, tuple[float, float]]:
        """Compute every named point position from the actuated angles."""
        ...

    @abstractmethod
    def inverse_kinematics(
        self, x: float, y: float, current_angles_rad: list[float]
    ) -> Optional[list[float]]:
        """Return actuated angles that place the EEF at (x, y), or *None*."""
        ...

    @abstractmethod
    def get_links(self) -> list[LinkDrawing]:
        """Visual specs for every link to draw."""
        ...

    @abstractmethod
    def get_points(self) -> list[PointDrawing]:
        """Visual specs for every point / joint marker to draw."""
        ...

    @abstractmethod
    def end_effector_name(self) -> str:
        """Name of the end-effector point (must appear in FK output)."""
        ...

    @abstractmethod
    def view_limits(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return ``((xmin, xmax), (ymin, ymax))`` for the default view."""
        ...

    # ── optional serial support (override to enable) ──────────────────

    def serial_config(self) -> dict:
        """Return ``{'baudrate': 9600, ...}`` to enable serial. Empty = off."""
        return {}

    def serial_command(self, joint_angles_deg: list[float]) -> bytes | None:
        """Build the bytes to send over serial, or *None* to skip."""
        return None
