"""
rrr -- 3-DOF serial robot.

This file shows how to implement ``RobotModel`` for a simple serial arm.
Unlike the parallel EEZYbotARM, an RRR chain has no loop-closure: the
symbolic points depend only on the actuated angles, so the default
``forward_kinematics`` / ``inverse_kinematics`` from the base classes
work directly. The symbolic model is built once (takes a moment) then
all runtime queries use fast lambdified numpy functions.
"""

import sympy
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame

from dataclasses import replace

from .symbolic import SymbolicRobotModel
from .model import (
    JointInfo, LinkDrawing, PointDrawing, ServoCal,
)


class rrr(SymbolicRobotModel):
    """3-DOF planar serial robot (three revolute joints in series)."""

    eef_name = "EEF1"
    serial_baudrate = 115200
    # Gripper is servo 7 on this arm, but not wired yet -- setting it enables
    # the inherited Open/Close actions and their LB/RB bindings.
    gripper_id = None

    #: One entry per actuated joint, in joint_info() order.  Per physical arm:
    #: re-measure after any horn is removed.  Defaults assume the servo reads
    #: 90 when the joint is at 0 and turns with the joint's positive sense.
    DEFAULT_SERVOS = [
        ServoCal(8,  q_ref=0, servo_ref=90),
        ServoCal(9,  q_ref=0, servo_ref=90),
        ServoCal(10, q_ref=0, servo_ref=90),
    ]

    #: Design (link lengths) plus per-unit build calibration.  Pass a partial
    #: dict to the constructor to describe a different physical arm.
    DEFAULT_PARAMS = {
        # Link lengths (m)
        "LA01": 80e-3,
        "LB01": 80e-3,
        "LC01": 40e-3,

        # Construction offsets: angle from the modelled frame axis to the
        # physical link centreline, deg, positive CCW.  Measure once from CAD
        # or the real arm.  All zero => theta_A from +x, theta_B/theta_C
        # relative with 0 = collinear with the previous link.
        "alpha_A_deg": 0,
        "alpha_B_deg": 0,
        "alpha_C_deg": 0,

        # Rotation sense of each joint angle: +1 = CCW positive (matches the
        # viewer and makes omega_C = (thA' + thB' + thC') z), -1 = CW positive,
        # for a joint whose useful travel is one-sided.
        "sign_A": 1,
        "sign_B": 1,
        "sign_C": 1,
    }

    #: Actuated DOFs: travel limits and home pose, in the convention fixed by
    #: the alphas and signs above.  Per-unit like the params -- mechanical hard
    #: stops move when the arm is rebuilt.
    DEFAULT_JOINTS = [
        JointInfo("Joint A", -90, 90, 0),
        JointInfo("Joint B", -90, 90, 0),
        JointInfo("Joint C", -90, 90, 0),
    ]

    def __init__(self, params=None, joints=None, servos=None):
        self._params = {**self.DEFAULT_PARAMS, **(params or {})}
        # Copy, so tweaking one arm's limits cannot leak into the next.
        self._joints = list(joints) if joints else [
            replace(j) for j in self.DEFAULT_JOINTS
        ]
        self.servo_cal = list(servos) if servos else [
            replace(s) for s in self.DEFAULT_SERVOS
        ]
        print("  Building symbolic model ...")
        self._build_model()
        print("  Done.")

    # ── symbolic build (runs once) ────────────────────────────────────────

    def _build_model(self):
        body_names = ["A", "B", "C"]
        theta = dynamicsymbols(
            " ".join(f"theta_{b}" for b in body_names)
        )
        N = ReferenceFrame("N")

        # Link-length symbols
        (LA01, LB01, LC01) = sympy.symbols("LA01 LB01 LC01")

        # Joint-convention symbols: each frame is placed at
        #     sign_X * theta_X + alpha_X_deg
        # so the joint angles used everywhere else -- limits, defaults,
        # actions, IK, sliders -- stay in one meaningful convention.
        (alpha_A_deg, alpha_B_deg, alpha_C_deg) = sympy.symbols(
            "alpha_A_deg alpha_B_deg alpha_C_deg"
        )
        (sign_A, sign_B, sign_C) = sympy.symbols("sign_A sign_B sign_C")

        # Symbol names match the keys of self._params.
        param_subs = {
            s: self._params[s.name]
            for s in (LA01, LB01, LC01,
                      alpha_A_deg, alpha_B_deg, alpha_C_deg,
                      sign_A, sign_B, sign_C)
        }

        # Serial chain: each frame rotates relative to the previous one,
        # so theta[i] are relative joint angles.
        def q(sign, theta_i, alpha_deg):
            return sign * theta_i + sympy.rad(alpha_deg)

        frames = {"A": N.orientnew(
            "A", "Axis", [q(sign_A, theta[0], alpha_A_deg), N.z])}
        frames["B"] = frames["A"].orientnew(
            "B", "Axis", [q(sign_B, theta[1], alpha_B_deg), N.z])
        frames["C"] = frames["B"].orientnew(
            "C", "Axis", [q(sign_C, theta[2], alpha_C_deg), N.z])

        # Named points (each link tip is the previous tip + link vector)
        pts = {}
        pts["O0"]   = 0 * N.x
        pts["A1"]   = pts["O0"] + LA01 * frames["A"].x
        pts["B1"]   = pts["A1"] + LB01 * frames["B"].x
        pts["C1"]   = pts["B1"] + LC01 * frames["C"].x
        pts["EEF1"] = pts["C1"]

        self._lambdify_points(theta, pts, N, param_subs)

        # Reach for view limits
        self._reach = sum(
            self._params[k] for k in ("LA01", "LB01", "LC01")
        )

    # ── RobotModel interface ──────────────────────────────────────────────
    #
    # forward_kinematics() and inverse_kinematics() are inherited:
    # a serial chain needs no loop-closure solve, so the defaults apply.

    def joint_info(self):
        return self._joints

    def get_links(self):
        return [
            LinkDrawing("O0", "A1", "#a6e3a1", 3),   # green
            LinkDrawing("A1", "B1", "#89b4fa", 3),   # blue
            LinkDrawing("B1", "C1", "#f38ba8", 3),   # red
        ]

    def get_points(self):
        return [
            PointDrawing("O0",   "s", "#cdd6f4", 8),    # base pivot
            PointDrawing("A1",   "o", "#bac2de", 5),
            PointDrawing("B1",   "o", "#bac2de", 5),
            PointDrawing("EEF1", "*", "#f38ba8", 14),   # end-effector
        ]

    def view_limits(self):
        r = self._reach * 1.15
        return ((-r, r), (-r, r))

    # ── actions ───────────────────────────────────────────────────────

    # actions() is inherited (returns none): the old waypoints predate the
    # alpha/sign convention.  Add them back once the home pose is measured.
    #
    # serial_actions() and button_map() are inherited too, and stay empty
    # while gripper_id is None -- set it to the gripper's servo id to get
    # open/close buttons and the LB/RB bindings.
    #
    # serial_command() is inherited: it maps each joint angle through
    # servo_cal and emits one "Y <id> <deg>" line per joint.
