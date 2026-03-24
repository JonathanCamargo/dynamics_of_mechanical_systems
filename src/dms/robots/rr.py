"""
RR -- 2-DOF serial revolute-revolute arm.

Implements ``RobotModel`` so it can be used with the robot viewer.
The symbolic model is built once then all runtime queries use fast
lambdified numpy functions.
"""

from sympy import symbols
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame

from .symbolic import SymbolicRobotModel
from .model import JointInfo, LinkDrawing, PointDrawing


class RR(SymbolicRobotModel):
    """2-DOF serial arm (two revolute joints)."""

    eef_name = "EEF"
    ik_tolerance = 0.02

    def __init__(self, params=None):
        if params is None:
            params = {"L1": 1, "L2": 1}
        self._params = params
        self._build_model()

    # ── symbolic build (runs once) ────────────────────────────────────────

    def _build_model(self):
        L1, L2 = symbols("L1 L2")
        theta1, theta2 = dynamicsymbols("theta1 theta2")
        self._theta = [theta1, theta2]

        N = ReferenceFrame("N")
        A = N.orientnew("A", "Axis", (theta1, N.z))
        B = A.orientnew("B", "Axis", (theta2, N.z))

        param_subs = {L1: self._params["L1"], L2: self._params["L2"]}

        r1 = A.x * L1
        r2 = B.x * L2

        # Named points
        pts = {}
        pts["O"]       = 0 * N.x
        pts["A"]       = r1
        pts["EEF"]     = r1 + r2
        pts["B"]       = r1 + 0.95 * r2
        pts["C_left"]  = pts["B"] - L1 * 0.05 * B.y
        pts["C_right"] = pts["B"] + L1 * 0.05 * B.y
        pts["D_left"]  = pts["C_left"]  + L1 * 0.075 * B.x
        pts["D_right"] = pts["C_right"] + L1 * 0.075 * B.x

        self._lambdify_points(self._theta, pts, N, param_subs)

        # Reach for view limits
        self._reach = self._params["L1"] + self._params["L2"]

    # ── RobotModel interface ──────────────────────────────────────────────

    def joint_info(self):
        return [
            JointInfo("Joint 1", -180, 180, 45),
            JointInfo("Joint 2", -180, 180, -45),
        ]

    def get_links(self):
        return [
            LinkDrawing("O",       "A",       "#89b4fa", 3),   # blue  - link 1
            LinkDrawing("A",       "B",       "#f38ba8", 3),   # red   - link 2
            LinkDrawing("B",       "C_left",  "#f38ba8", 2),   # gripper
            LinkDrawing("B",       "C_right", "#f38ba8", 2),
            LinkDrawing("C_left",  "D_left",  "#f38ba8", 2),
            LinkDrawing("C_right", "D_right", "#f38ba8", 2),
        ]

    def get_points(self):
        return [
            PointDrawing("O",   "s", "#cdd6f4", 8),    # base pivot
            PointDrawing("A",   "o", "#bac2de", 6),     # elbow
            PointDrawing("EEF", "*", "#f38ba8", 14),    # end-effector
        ]

    def view_limits(self):
        r = self._reach * 1.15
        return ((-r, r), (-r, r))
