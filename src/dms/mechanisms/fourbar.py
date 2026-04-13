from typing import Any, ClassVar

import sympy
from sympy import symbols
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame
from .. import getComponents
import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import animation


def Animate(fourbar, trajectory_star=None):
    fig, ax = plt.subplots()
    trajectories, thetas = GetTrajectory(fourbar)
    all_markers = np.vstack(list(trajectories.values()))
    min_lims = np.nanmin(all_markers, axis=0)
    max_lims = np.nanmax(all_markers, axis=0)
    center = (min_lims + max_lims) / 2
    range = (max_lims - min_lims)
    marker_keys = list(trajectories.keys())

    def update(frame):
        ax.cla()
        fourbar.plot(thetas[frame, 0], ax=ax, theta2=thetas[frame, 1], theta3=thetas[frame, 2])
        for j, k in enumerate(marker_keys):
            c = fourbar._bar_colors[fourbar._marker_bars[j]]
            traj = trajectories[k]
            ax.plot(traj[0:frame+1, 0], traj[0:frame+1, 1], '-', color=c)
            ax.plot(traj[frame, 0], traj[frame, 1], '*', color=c)
        if trajectory_star is not None:
            ax.plot(trajectory_star[:, 0], trajectory_star[:, 1], 'b.')
        ax.set_xlim(center[0]-2*range.max(), center[0]+2*range.max())
        ax.set_ylim(center[1]-2*range.max(), center[1]+2*range.max())
        return ()

    ani = animation.FuncAnimation(fig, update, frames=thetas.shape[0], interval=50)
    return ani, fig


def _marker_offset_key(px, py):
    """Cache key for a marker offset; numeric and symbolic offsets get
    distinct slots so that a baked-in literal and an expression like
    ``L2/2`` never collide.
    """
    if isinstance(px, sympy.Expr) or isinstance(py, sympy.Expr):
        return ('sym', sympy.srepr(sympy.sympify(px)), sympy.srepr(sympy.sympify(py)))
    try:
        return ('num', float(px), float(py))
    except (TypeError, ValueError):
        return ('sym', sympy.srepr(sympy.sympify(px)), sympy.srepr(sympy.sympify(py)))


def GetTrajectory(fourbar, n_points=40):
    theta_array = np.linspace(0+0.4, 2*np.pi+0.4, n_points)
    marker_keys = list(fourbar._marker_funs.keys())
    trajectories = {k: np.nan*np.ones((len(theta_array), 2)) for k in marker_keys}
    thetas = np.nan*np.ones((len(theta_array), 3))
    for i in range(len(theta_array)):
        theta1 = theta_array[i]
        [theta2, theta3], fkout = fourbar.FK(theta1)
        if fkout.cost > 1e-3:
            break
        points = fourbar.ComputePoints(theta1, theta2, theta3)
        for k in marker_keys:
            trajectories[k][i, :] = points[k]
        thetas[i, :] = [theta1, theta2, theta3]
    return trajectories, thetas


# fourbar
class FourBar:
    _cache: ClassVar[dict[str, Any]] = {}
    _grad_cache: ClassVar[dict[str, Any]] = {}
    _marker_cache: ClassVar[dict[Any, Any]] = {}

    @classmethod
    def length_symbols(cls):
        """Return the symbolic link-length variables ``(L0, L1, L2, L3)``.

        Use these to build marker offsets that *track* the design parameters,
        e.g. ``markers=[(2, (L2/2, 0))]``. Such markers update with
        ``set_lengths`` and contribute the direct ``∂offset/∂l`` term to
        ``gradient``. Numeric (float) marker offsets keep the old, fixed
        behavior.
        """
        cls._ensure_cache()
        sym = cls._cache['sym']
        return sym['L0'], sym['L1'], sym['L2'], sym['L3']

    @classmethod
    def _ensure_cache(cls):
        if cls._cache:
            return

        theta1, theta2, theta3 = dynamicsymbols('theta1 theta2 theta3')
        L0, L1, L2, L3 = symbols('l0 l1 l2 l3')

        N = ReferenceFrame('N')
        A = N.orientnew('A', 'Axis', [theta1, N.z])
        B = N.orientnew('B', 'Axis', [theta2, N.z])
        C = N.orientnew('C', 'Axis', [theta3, N.z])

        r0 = L0 * N.x
        r1 = L1 * A.x
        r2 = L2 * B.x
        r3 = L3 * C.x

        eqLoop = r0 + r3 - r1 - r2

        points = {
            'O': 0*N.x, 'A': r1, 'B': r1 + r2,
            'Bprime': r0 + r3, 'C': r0,
        }

        args = [theta1, theta2, theta3, L0, L1, L2, L3]

        pos_fun = sympy.lambdify(args, getComponents(eqLoop, N)[0:-1])
        points_fun = {
            k: sympy.lambdify(args, getComponents(v, N)[0:-1])
            for k, v in points.items()
        }

        bar_origins = {
            0: 0*N.x,  # ground: O
            1: 0*N.x,  # crank: O
            2: r1,     # coupler: A
            3: r0,     # follower: C
        }
        bar_frames = {
            0: N,
            1: A,
            2: B,
            3: C,
        }

        cls._cache = {
            'pos_fun': pos_fun,
            'points_fun': points_fun,
            'args': args,
            'sym': {
                'theta1': theta1, 'theta2': theta2, 'theta3': theta3,
                'L0': L0, 'L1': L1, 'L2': L2, 'L3': L3,
                'N': N, 'A': A, 'B': B, 'C': C,
                'r0': r0, 'r1': r1, 'r2': r2, 'r3': r3,
                'points': points,
                'eqLoop': eqLoop,
                'bar_origins': bar_origins,
                'bar_frames': bar_frames,
            },
        }

    @classmethod
    def _ensure_grad_cache(cls):
        if cls._grad_cache:
            return
        cls._ensure_cache()

        sym = cls._cache['sym']
        N = sym['N']
        args = cls._cache['args']
        eqLoop = sym['eqLoop']
        points = sym['points']

        theta2, theta3 = sym['theta2'], sym['theta3']
        L0, L1, L2, L3 = sym['L0'], sym['L1'], sym['L2'], sym['L3']

        passive = [theta2, theta3]
        design = [L0, L1, L2, L3]

        loop_comps = getComponents(eqLoop, N, simplify=False)[0:-1]

        dF_dtheta = sympy.Matrix([
            [sympy.diff(f, t) for t in passive] for f in loop_comps
        ])
        dF_dl = sympy.Matrix([
            [sympy.diff(f, l) for l in design] for f in loop_comps
        ])

        point_jacs = {}
        for k, v in points.items():
            comps = getComponents(v, N, simplify=False)[0:-1]
            dP_dtheta = sympy.Matrix([
                [sympy.diff(c, t) for t in passive] for c in comps
            ])
            dP_dl = sympy.Matrix([
                [sympy.diff(c, l) for l in design] for c in comps
            ])
            point_jacs[k] = {
                'dP_dtheta': sympy.lambdify(args, dP_dtheta),
                'dP_dl': sympy.lambdify(args, dP_dl),
            }

        cls._grad_cache = {
            'dF_dtheta': sympy.lambdify(args, dF_dtheta),
            'dF_dl': sympy.lambdify(args, dF_dl),
            'point_jacs': point_jacs,
        }

    @classmethod
    def _get_marker_funs(cls, markers_key, markers):
        if markers_key in cls._marker_cache:
            return cls._marker_cache[markers_key]

        cls._ensure_cache()
        sym = cls._cache['sym']
        N = sym['N']
        args = cls._cache['args']
        bar_origins = sym['bar_origins']
        bar_frames = sym['bar_frames']

        marker_funs = {}
        for i, (bar_idx, (px, py)) in enumerate(markers):
            origin = bar_origins[bar_idx]
            frame = bar_frames[bar_idx]
            marker_expr = origin + px * frame.x + py * frame.y
            name = f'marker_{i + 1}'
            marker_funs[name] = sympy.lambdify(
                args, getComponents(marker_expr, N)[0:-1]
            )

        cls._marker_cache[markers_key] = marker_funs
        return marker_funs

    @classmethod
    def _get_marker_grad_funs(cls, markers_key, markers):
        grad_key = ('grad', markers_key)
        if grad_key in cls._marker_cache:
            return cls._marker_cache[grad_key]

        cls._ensure_cache()
        cls._ensure_grad_cache()
        sym = cls._cache['sym']
        N = sym['N']
        args = cls._cache['args']
        bar_origins = sym['bar_origins']
        bar_frames = sym['bar_frames']

        theta2, theta3 = sym['theta2'], sym['theta3']
        L0, L1, L2, L3 = sym['L0'], sym['L1'], sym['L2'], sym['L3']
        passive = [theta2, theta3]
        design = [L0, L1, L2, L3]

        marker_grads = {}
        for i, (bar_idx, (px, py)) in enumerate(markers):
            origin = bar_origins[bar_idx]
            frame = bar_frames[bar_idx]
            marker_expr = origin + px * frame.x + py * frame.y
            comps = getComponents(marker_expr, N, simplify=False)[0:-1]

            dP_dtheta = sympy.Matrix([
                [sympy.diff(c, t) for t in passive] for c in comps
            ])
            dP_dl = sympy.Matrix([
                [sympy.diff(c, l) for l in design] for c in comps
            ])

            name = f'marker_{i + 1}'
            marker_grads[name] = {
                'dP_dtheta': sympy.lambdify(args, dP_dtheta),
                'dP_dl': sympy.lambdify(args, dP_dl),
            }

        cls._marker_cache[grad_key] = marker_grads
        return marker_grads

    def __init__(self, l0, l1, l2, l3, markers=None):
        # A four bar mechanism
        # l0: fixed link assumed horizontal. Use setRot/setOloc for absolute reference
        # l1 is the driving link, driven by theta1
        # markers: list of (bar_index, (dx, dy)) tuples. bar_index 0-3, (dx,dy) in bar's local frame.
        #          dx/dy may be floats (fixed offsets) or sympy expressions in the
        #          length symbols from FourBar.length_symbols() — symbolic offsets
        #          track set_lengths() and contribute to gradient().
        #          If None, defaults to symbolic midpoint of coupler bar: [(2, (L2/2, 0))]
        self.__class__._ensure_cache()
        self.lengths = [l0, l1, l2, l3]
        self.zpos = [0.1, 0.1]
        self.oloc = np.array([0, 0])
        self.rotm = np.eye(2)

        if markers is None:
            _, _, L2, _ = self.__class__.length_symbols()
            markers = [(2, (L2/2, 0))]
        self._markers = markers
        self._marker_bars = [bar for bar, _ in markers]
        self._markers_key = tuple(
            (b, _marker_offset_key(px, py)) for b, (px, py) in markers
        )
        self._marker_funs = self.__class__._get_marker_funs(
            self._markers_key, self._markers
        )

    def set_lengths(self, l0=None, l1=None, l2=None, l3=None):
        """Update link lengths without rebuilding symbolic expressions."""
        if l0 is not None:
            self.lengths[0] = l0
        if l1 is not None:
            self.lengths[1] = l1
        if l2 is not None:
            self.lengths[2] = l2
        if l3 is not None:
            self.lengths[3] = l3

    def setOloc(self, x, y):
        self.oloc = np.array([x, y])

    def setRotm(self, rotm):
        self.rotm = rotm

    def setRot(self, theta):
        self.rotm = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ])

    def ComputePoints(self, theta1, theta2=None, theta3=None):
        if theta2 is None or theta3 is None:
            z, _ = self.FK(theta1)
            theta2, theta3 = z
        lengths = self.lengths
        points_fun = self._cache['points_fun']
        point_vals = {
            k: np.matmul(self.rotm, np.array(point(theta1, theta2, theta3, *lengths))) + self.oloc
            for k, point in points_fun.items()
        }
        for name, fun in self._marker_funs.items():
            point_vals[name] = (
                np.matmul(self.rotm, np.array(fun(theta1, theta2, theta3, *lengths)))
                + self.oloc
            )
        return point_vals

    def plot(self, theta1, ax=None, theta2=None, theta3=None):
        if ax is None:
            ax = plt.gca()
        point_vals = self.ComputePoints(theta1, theta2, theta3)
        for k, p in point_vals.items():
            if k.startswith('marker_'):
                continue
            ax.plot(p[0], p[1], 'ko')
        self._bar_colors = {}
        self._bar_colors[0] = 'k'
        bar_colors = self._bar_colors
        ax.plot([point_vals['O'][0], point_vals['C'][0]], [point_vals['O'][1], point_vals['C'][1]], 'k')
        bar_colors[1] = ax.plot([point_vals['O'][0], point_vals['A'][0]], [point_vals['O'][1], point_vals['A'][1]])[0].get_color()
        bar_colors[2] = ax.plot([point_vals['A'][0], point_vals['B'][0]], [point_vals['A'][1], point_vals['B'][1]])[0].get_color()
        ax.plot([point_vals['B'][0], point_vals['Bprime'][0]], [point_vals['B'][1], point_vals['Bprime'][1]], 'k:')
        bar_colors[3] = ax.plot([point_vals['Bprime'][0], point_vals['C'][0]], [point_vals['Bprime'][1], point_vals['C'][1]])[0].get_color()
        bar_endpoints = {0: ('O', 'C'), 1: ('O', 'A'), 2: ('A', 'B'), 3: ('C', 'Bprime')}
        for j, bar in enumerate(self._marker_bars):
            k = f'marker_{j+1}'
            s, e = bar_endpoints[bar]
            mid = (point_vals[s] + point_vals[e]) / 2
            c = bar_colors[bar]
            ax.plot([mid[0], point_vals[k][0]], [mid[1], point_vals[k][1]], '-', color=c)
            ax.plot(point_vals[k][0], point_vals[k][1], '*', color=c, markersize=10)

    def FK(self, theta1, zpos=None):
        if zpos is None:
            zpos = self.zpos
        lengths = self.lengths
        pos_fun = self._cache['pos_fun']
        out = scipy.optimize.least_squares(
            lambda x: pos_fun(theta1, *x, *lengths), zpos
        )
        kThreshold = 1e-3
        if out.cost < kThreshold:
            self.zpos = out.x
        return out.x, out

    def grashof(self):
        """Check Grashof's condition: shortest + longest <= sum of remaining two."""
        s = sorted(self.lengths)
        return s[0] + s[3] <= s[1] + s[2]

    def grashof_margin(self):
        """Distance from the Grashof boundary; positive means satisfied."""
        s = sorted(self.lengths)
        return (s[1] + s[2]) - (s[0] + s[3])

    def gradient(self, point_name, theta1, theta2, theta3, wrt=None):
        """Gradient of a point position w.r.t. link lengths via the implicit
        function theorem on the loop closure equation:
            dtheta/dl = -(dF/dtheta)^{-1} dF/dl
            dP/dl    = dP/dtheta @ dtheta/dl + dP/dl_direct

        Parameters
        ----------
        point_name : str
            Name of point ('O', 'A', 'B', 'Bprime', 'C', 'marker_1', ...).
        theta1, theta2, theta3 : float
            Crank, coupler and follower angles (theta2/theta3 from FK).
        wrt : list of int, optional
            Link indices to differentiate w.r.t.  Default ``[2, 3]``.

        Returns
        -------
        ndarray, shape (2, len(wrt))
        """
        if wrt is None:
            wrt = [2, 3]

        self.__class__._ensure_grad_cache()
        gc = self._grad_cache
        lengths = self.lengths
        call_args = (theta1, theta2, theta3, *lengths)

        dF_dtheta = np.array(gc['dF_dtheta'](*call_args), dtype=float)
        dF_dl = np.array(gc['dF_dl'](*call_args), dtype=float)

        dtheta_dl = -np.linalg.solve(dF_dtheta, dF_dl)  # (2, 4)

        if point_name.startswith('marker_'):
            marker_grads = self.__class__._get_marker_grad_funs(
                self._markers_key, self._markers
            )
            jac = marker_grads[point_name]
        else:
            jac = gc['point_jacs'][point_name]

        dP_dtheta = np.array(jac['dP_dtheta'](*call_args), dtype=float)
        dP_dl = np.array(jac['dP_dl'](*call_args), dtype=float)

        dP_dl_total = dP_dtheta @ dtheta_dl + dP_dl  # (2, 4)
        return dP_dl_total[:, wrt]
