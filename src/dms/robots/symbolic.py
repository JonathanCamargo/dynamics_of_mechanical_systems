"""
Sympy convenience layer for ``RobotModel``.

Subclass ``SymbolicRobotModel`` when your kinematics are built with
``sympy.physics.mechanics``.  Call ``_lambdify_points()`` in your
``__init__`` and ``forward_kinematics()`` is provided for free.

If your mechanism has extra internal coordinates (e.g. loop-closure),
override ``forward_kinematics()`` and call ``self._pts_fn`` yourself
-- see ``EEZYbotARM`` for an example.
"""

from .model import RobotModel


class SymbolicRobotModel(RobotModel):
    """``RobotModel`` with built-in sympy → numpy lambdification.

    How to use
    ----------
    1. Build your symbolic mechanism (frames, points, params).
    2. Call ``self._lambdify_points(theta, pts, N, params)`` once.
    3. ``forward_kinematics()`` now works automatically.

    If your robot has loop-closure or other internal DOFs, override
    ``forward_kinematics()`` and use ``self._pts_fn`` directly.
    """

    def _lambdify_points(self, theta, pts, N, params):
        """Lambdify a dict of sympy position vectors into fast callables.

        Parameters
        ----------
        theta : list of dynamicsymbols
            The generalized coordinates (actuated + internal).
        pts : dict[str, sympy Vector]
            Named point positions expressed in frame *N*.
        N : ReferenceFrame
            The inertial (base) reference frame.
        params : dict
            Numeric substitutions for all symbolic parameters.
        """
        import sympy
        self._pts_fn = {}
        for name, vec in pts.items():
            x_expr = vec.dot(N.x).subs(params)
            y_expr = vec.dot(N.y).subs(params)
            self._pts_fn[name] = sympy.lambdify(
                theta, [x_expr, y_expr], "numpy"
            )

    def forward_kinematics(self, joint_angles_rad):
        """Default FK: evaluate the lambdified point functions.

        This works directly for serial chains.  Override for parallel
        mechanisms that need loop-closure solves first.
        """
        return {
            name: tuple(fn(*joint_angles_rad))
            for name, fn in self._pts_fn.items()
        }
