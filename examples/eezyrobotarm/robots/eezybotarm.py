"""
EEZYbotARM -- 2-DOF parallel linkage mechanism.

This file shows how to implement ``RobotModel`` for a real robot.
The symbolic model is built once (takes a few seconds) then all
runtime queries use fast lambdified numpy functions.
"""

import numpy as np
import sympy
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame
from scipy.optimize import fsolve, least_squares

from robot_viewer.model import RobotModel, JointInfo, LinkDrawing, PointDrawing


class EEZYbotARM(RobotModel):
    """2-DOF parallel mechanism (8 bodies, 6 loop-closure constraints)."""

    def __init__(self):
        print("  Building symbolic model ...")
        self._build_model()
        self._z0 = np.deg2rad([90, -90, 0, 0, 0, 100])
        print("  Done.")

    # ── symbolic build (runs once) ────────────────────────────────────────

    def _build_model(self):
        body_names = ["A", "B", "C", "D", "E", "F", "G", "H"]
        theta = dynamicsymbols(
            " ".join(f"theta_{b}" for b in body_names)
        )
        N = ReferenceFrame("N")
        frames = {
            b: N.orientnew(b, "Axis", [theta[i], N.z])
            for i, b in enumerate(body_names)
        }

        # Length symbols
        (LA01, LB01, LC01, LD01, LG01, LH01) = sympy.symbols(
            "LA01 LB01 LC01 LD01 LG01 LH01"
        )
        (LE01, LE02, LF01, LF02x, LF02y) = sympy.symbols(
            "LE01 LE02 LF01 LF02x LF02y"
        )
        (L001x, L001y, LEEF01x, LEEF01y) = sympy.symbols(
            "L001x L001y LEEF01x LEEF01y"
        )

        params = {
            LA01: 80e-3,  LB01: 35e-3,  LC01: 80e-3,   LD01: 80e-3,
            LG01: 80e-3,  LH01: 20e-3,  LE01: 35e-3,   LE02: 80e-3,
            LF01: 46e-3,  LF02x: 36e-3, LF02y: 17e-3,
            L001x: 36e-3, L001y: 17e-3,
            LEEF01x: 17e-3, LEEF01y: 30e-3,
        }

        # Named points
        pts = {}
        pts["O0"]   = 0 * N.x
        pts["O1"]   = L001x * N.x + L001y * N.y
        pts["A1"]   = LA01 * frames["A"].x
        pts["B1"]   = LB01 * frames["B"].x
        pts["C1"]   = pts["B1"] + LC01 * frames["C"].x
        pts["E0"]   = pts["C1"] - LE01 * frames["E"].x
        pts["E2"]   = pts["C1"] - (LE01 + LE02) * frames["E"].x
        pts["H1"]   = pts["E2"] + LH01 * frames["H"].x
        pts["G1"]   = pts["H1"] + LG01 * frames["G"].x
        pts["F1"]   = pts["G1"] + LF01 * frames["F"].x
        pts["EEF1"] = (
            pts["E2"]
            + LEEF01x * frames["H"].x
            + LEEF01y * frames["H"].y
        )

        # Lambdify every point -> callable(theta_A .. theta_H) -> [x, y]
        self._pts_fn = {}
        for name, vec in pts.items():
            x_expr = vec.dot(N.x).subs(params)
            y_expr = vec.dot(N.y).subs(params)
            self._pts_fn[name] = sympy.lambdify(theta, [x_expr, y_expr], "numpy")

        # Loop-closure equations  (6 scalar eqs for 6 unknowns)
        eqL1 = (
            pts["E0"]
            + LF02x * frames["F"].x
            + LF02y * frames["F"].y
            + LD01 * frames["D"].x
            - pts["O1"]
        )
        eqL2 = pts["E0"] - pts["A1"]
        eqL3 = (
            LH01 * frames["H"].x
            + LG01 * frames["G"].x
            + (LF01 - LF02x) * frames["F"].x
            - LF02y * frames["F"].y
            - LE02 * frames["E"].x
        )

        eq_scalars = []
        for eq_vec in (eqL1, eqL2, eqL3):
            eq_scalars.append(eq_vec.dot(N.x))
            eq_scalars.append(eq_vec.dot(N.y))
        eq_scalars = [e.subs(params) for e in eq_scalars]

        self._eq_fn = sympy.lambdify(theta, eq_scalars, "numpy")

        # Joint limits  (bounds for least_squares in IK)
        self._limits = (np.deg2rad([45, -60]), np.deg2rad([135, 90]))

    # ── internal FK solve ─────────────────────────────────────────────────

    def _solve_loop(self, thA, thB, z0=None):
        if z0 is None:
            z0 = self._z0.copy()
        return fsolve(lambda z: self._eq_fn(thA, thB, *z), z0)

    # ── RobotModel interface ──────────────────────────────────────────────

    def joint_info(self):
        return [
            JointInfo("Motor A", 45, 135, 90),
            JointInfo("Motor B", -60,  90, 30),
        ]

    def forward_kinematics(self, joint_angles_rad):
        thA, thB = joint_angles_rad
        z = self._solve_loop(thA, thB, self._z0)
        self._z0 = z  # warm-start next call
        all_th = (thA, thB, *z)
        return {name: tuple(fn(*all_th)) for name, fn in self._pts_fn.items()}

    def inverse_kinematics(self, x, y, current_angles_rad):
        z0 = self._z0.copy()
        x0 = np.array(current_angles_rad[:2])

        def residual(qpos):
            z = self._solve_loop(*qpos, z0=z0)
            eef = self._pts_fn["EEF1"](*qpos, *z)
            # Return error in mm (scalar → 1-element residual vector)
            return np.linalg.norm(np.array(eef) - np.array([x, y])) * 1000

        sln = least_squares(residual, x0, bounds=self._limits)
        if sln.fun[0] > 5.0:         # reject if EEF error > 5 mm
            return None
        return list(sln.x)

    def get_links(self):
        return [
            LinkDrawing("O0",  "A1",   "#a6e3a1", 3),   # green
            LinkDrawing("O0",  "B1",   "#94e2d5", 3),   # teal
            LinkDrawing("B1",  "C1",   "#f38ba8", 3),   # red
            LinkDrawing("C1",  "E2",   "#89b4fa", 3),   # blue
            LinkDrawing("E2",  "H1",   "#fab387", 3),   # peach
            LinkDrawing("E2",  "EEF1", "#fab387", 2),   # peach (EEF branch)
            LinkDrawing("H1",  "G1",   "#cba6f7", 3),   # mauve
            LinkDrawing("G1",  "F1",   "#f9e2af", 3),   # yellow
            LinkDrawing("G1",  "E0",   "#f9e2af", 2),   # yellow
            LinkDrawing("F1",  "E0",   "#f9e2af", 2),   # yellow
            LinkDrawing("F1",  "O1",   "#f5c2e7", 3),   # pink
        ]

    def get_points(self):
        return [
            PointDrawing("O0",   "s", "#cdd6f4", 8),   # base pivot
            PointDrawing("O1",   "s", "#cdd6f4", 8),   # base pivot
            PointDrawing("A1",   "o", "#bac2de", 5),
            PointDrawing("B1",   "o", "#bac2de", 5),
            PointDrawing("C1",   "o", "#bac2de", 5),
            PointDrawing("E0",   "o", "#bac2de", 5),
            PointDrawing("E2",   "o", "#bac2de", 5),
            PointDrawing("H1",   "o", "#bac2de", 5),
            PointDrawing("G1",   "o", "#bac2de", 5),
            PointDrawing("F1",   "o", "#bac2de", 5),
            PointDrawing("EEF1", "*", "#f38ba8", 14),   # end-effector
        ]

    def end_effector_name(self):
        return "EEF1"

    def view_limits(self):
        return ((-0.18, 0.15), (-0.10, 0.20))

    # ── serial ────────────────────────────────────────────────────────

    def serial_config(self):
        return {"baudrate": 115200}

    def serial_command(self, joint_angles_deg):
        a0, a1 = joint_angles_deg
        # Servo 5 (Motor A): offset → 90 deg kinematic = 110 servo
        servo_0 = int(round(a0 + 20))
        # Servo 6 (Motor B): negated + offset → 0 deg kinematic = 85 servo
        servo_1 = int(round(85 - a1))
        return f"Y 5 {servo_0}\r\nY 6 {servo_1}\r\n".encode()
