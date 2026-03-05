"""
Workspace analysis and obstacle utilities for 2-D planar robots.
"""

from dataclasses import dataclass
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.path import Path


# ── Obstacles ─────────────────────────────────────────────────────────────────

@dataclass
class CircleObstacle:
    """Circular obstacle defined by centre and radius."""
    cx: float
    cy: float
    radius: float

    def contains(self, x, y):
        return (x - self.cx) ** 2 + (y - self.cy) ** 2 <= self.radius ** 2


@dataclass
class PolygonObstacle:
    """Polygonal obstacle defined by ordered vertices."""
    vertices: list  # list of (x, y) tuples

    def __post_init__(self):
        self._path = Path(self.vertices)

    def contains(self, x, y):
        return self._path.contains_point((x, y))


# ── Workspace computation ─────────────────────────────────────────────────────

def compute_workspace(robot, n=100):
    """Sweep all joints and return reachable EEF positions.

    Parameters
    ----------
    robot : RobotModel
        Robot implementing the RobotModel interface.
    n : int
        Number of samples per joint (total points = n ** num_joints).

    Returns
    -------
    xs, ys : ndarray
        End-effector x and y coordinates.
    """
    joints = robot.joint_info()
    ranges = [
        np.linspace(np.deg2rad(j.min_deg), np.deg2rad(j.max_deg), n)
        for j in joints
    ]
    grids = np.meshgrid(*ranges)
    combos = np.column_stack([g.ravel() for g in grids])

    eef_name = robot.end_effector_name()
    xs, ys = [], []
    for angles in combos:
        pts = robot.forward_kinematics(list(angles))
        pos = pts[eef_name]
        xs.append(pos[0])
        ys.append(pos[1])
    return np.array(xs), np.array(ys)


# ── Drawing helpers ───────────────────────────────────────────────────────────

def add_obstacles(ax, obstacles):
    """Draw obstacles on a matplotlib Axes."""
    for obs in obstacles:
        if isinstance(obs, CircleObstacle):
            patch = mpatches.Circle(
                (obs.cx, obs.cy), obs.radius,
                fill=True, facecolor="#f38ba8", edgecolor="#fab387",
                alpha=0.35, linewidth=1.5, zorder=1,
            )
            ax.add_patch(patch)
        elif isinstance(obs, PolygonObstacle):
            patch = mpatches.Polygon(
                obs.vertices, closed=True,
                fill=True, facecolor="#f38ba8", edgecolor="#fab387",
                alpha=0.35, linewidth=1.5, zorder=1,
            )
            ax.add_patch(patch)


# ── Collision detection ───────────────────────────────────────────────────────

def check_collision(x, y, obstacles):
    """Return True if point (x, y) is inside any obstacle."""
    return any(obs.contains(x, y) for obs in obstacles)
