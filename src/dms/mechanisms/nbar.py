"""General N-bar planar linkage mechanism with automatic kinematic analysis.

Unlike FourBar and FiveBar which hardcode kinematic equations for specific
topologies, NBarMechanism derives loop-closure equations automatically from
any topology graph, compiles them to fast numerical solvers with analytical
Jacobians, and handles singularities and branch tracking during crank sweeps.

Supports any closed-loop single-DOF planar linkage: 4-bar, 6-bar (Watt, Stephenson),
8-bar (16 topologies), 10-bar (230+ topologies), and custom topologies.

Usage::

    from dms.mechanisms.nbar import NBarMechanism, Animate, GetTrajectory

    # 4-bar from edge list
    mech = NBarMechanism(
        edges=[(0,1), (1,2), (2,3), (3,0)],
        link_lengths=[2.0, 1.0, 1.5, 1.0],
    )

    # Or from a networkx graph
    import networkx as nx
    G = nx.Graph([(0,1), (1,2), (2,3), (3,0)])
    mech = NBarMechanism(G, link_lengths=[2.0, 1.0, 1.5, 1.0])

    # Same workflow as FourBar / FiveBar
    solution, info = mech.FK(theta1=0.5)
    points = mech.ComputePoints(theta1=0.5)
    trajectory, thetas = GetTrajectory(mech, n_points=360)
    ani, fig = Animate(mech, trajectory_star=target_curve)
"""

import logging
import re
from dataclasses import dataclass

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy
from matplotlib import animation
from scipy.optimize import fsolve
from sympy import Matrix, Symbol, cos, lambdify, sin
from sympy.physics.mechanics import dynamicsymbols

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Loop-closure equation derivation
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class _LoopClosureResult:
    """Symbolic loop-closure equations for a planar mechanism."""

    equations: list
    unknowns: list
    parameters: list
    input_angle: object
    input_link: int
    loops: list


def _find_input_link(G):
    """First binary (degree-2) neighbor of the ground node (node 0)."""
    for nb in sorted(G.neighbors(0)):
        if G.degree(nb) == 2:
            return nb
    return sorted(G.neighbors(0))[0]


def _derive_loop_closure(G):
    """Derive symbolic loop-closure equations from a mechanism topology graph.

    Each node is a link (node 0 = ground), each edge is a revolute joint.
    Uses ``nx.cycle_basis`` to find independent loops and writes vector
    closure equations in trigonometric form.

    Parameters
    ----------
    G : nx.Graph
        Mechanism topology graph.

    Returns
    -------
    _LoopClosureResult
    """
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    n_loops = n_edges - n_nodes + 1

    # Angle symbols (ground angle = 0)
    angles = {0: 0}
    moving = {}
    for node in sorted(G.nodes()):
        if node == 0:
            continue
        theta = dynamicsymbols(f"theta{node}")
        angles[node] = theta
        moving[node] = theta

    # Length symbols
    lengths = {node: Symbol(f"l{node}", positive=True) for node in sorted(G.nodes())}

    # Input link
    input_link = _find_input_link(G)
    input_angle = moving[input_link]

    unknowns = [moving[n] for n in sorted(moving) if n != input_link]
    parameters = [lengths[n] for n in sorted(lengths)] + [input_angle]

    # Independent loops
    loops = sorted(nx.cycle_basis(G), key=len)[:n_loops]

    equations = []
    for loop in loops:
        eq_x, eq_y = 0, 0
        n = len(loop)
        for i in range(n):
            node = loop[i]
            pred = loop[(i - 1) % n]
            succ = loop[(i + 1) % n]
            direction = 1 if pred < succ else -1
            l = lengths[node]
            theta = angles[node]
            if theta == 0:
                eq_x += direction * l
            else:
                eq_x += direction * l * cos(theta)
                eq_y += direction * l * sin(theta)
        equations.extend([eq_x, eq_y])

    return _LoopClosureResult(
        equations=equations,
        unknowns=unknowns,
        parameters=parameters,
        input_angle=input_angle,
        input_link=input_link,
        loops=loops,
    )


# ═══════════════════════════════════════════════════════════════════════
# Compiled numerical solver
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class _CompiledSolver:
    """Fast numerical solver compiled from symbolic equations."""

    f_compiled: object
    J_compiled: object
    unknown_names: list
    parameter_names: list


def _compile_solver(loop_result):
    """Compile symbolic loop-closure equations to fast NumPy callables.

    Uses ``sympy.lambdify`` with common-subexpression elimination for
    both the equation vector and its analytical Jacobian.
    """
    eqs = loop_result.equations
    unknowns = loop_result.unknowns
    params = loop_result.parameters

    eq_vec = Matrix(eqs)
    J = eq_vec.jacobian(Matrix(unknowns))
    args = list(unknowns) + list(params)

    f_compiled = lambdify(args, eq_vec, modules="numpy", cse=True)
    J_compiled = lambdify(args, J, modules="numpy", cse=True)

    return _CompiledSolver(
        f_compiled=f_compiled,
        J_compiled=J_compiled,
        unknown_names=[str(u) for u in unknowns],
        parameter_names=[str(p) for p in params],
    )


# ═══════════════════════════════════════════════════════════════════════
# Position analysis (solve + initial solution search)
# ═══════════════════════════════════════════════════════════════════════


def _solve_position(solver, theta_input, params, guess):
    """Solve position analysis at one crank angle via fsolve + analytical Jacobian.

    Returns ``(solution, converged)``."""
    param_vals = list(params) + [theta_input]

    def residual(unknowns):
        return np.array(
            solver.f_compiled(*list(unknowns), *param_vals), dtype=float
        ).flatten()

    def jacobian(unknowns):
        J = np.array(
            solver.J_compiled(*list(unknowns), *param_vals), dtype=float
        )
        return J.reshape(len(guess), -1) if J.ndim == 1 else J

    sol, info, ier, msg = fsolve(residual, guess, fprime=jacobian, full_output=True)
    if ier == 1:
        converged = bool(np.max(np.abs(residual(sol))) < 1e-8)
    else:
        converged = False
    return sol, converged


def _find_initial_solution(solver, theta_input, params, n_tries=50):
    """Bootstrap an initial solution via random restarts in [-pi, pi]."""
    n_unknowns = len(solver.unknown_names)
    for _ in range(n_tries):
        guess = np.random.uniform(-np.pi, np.pi, size=n_unknowns)
        sol, ok = _solve_position(solver, theta_input, params, guess)
        if ok:
            return sol
    return None


# ═══════════════════════════════════════════════════════════════════════
# Branch-jump detection and recovery
# ═══════════════════════════════════════════════════════════════════════


def _detect_jump(prev, curr, max_change=np.radians(15)):
    """True if any angle changed by more than *max_change* between steps."""
    diff = (curr - prev + np.pi) % (2 * np.pi) - np.pi
    return bool(np.any(np.abs(diff) > max_change))


def _recover_branch(solver, theta, params, prev, n_sub=5):
    """Attempt to recover the current branch after a detected jump.

    Strategy 1: subdivide the crank step into smaller increments.
    Strategy 2: perturbed initial guesses around previous solution.
    """
    best, recovered = prev.copy(), False
    current = prev.copy()

    # Subdivision
    for sub_theta in np.linspace(theta - np.radians(1), theta, n_sub + 1)[1:]:
        sol, ok = _solve_position(solver, sub_theta, params, current)
        if ok and not _detect_jump(current, sol):
            current = sol
            best = sol
            recovered = True
        else:
            break

    if recovered:
        sol, ok = _solve_position(solver, theta, params, best)
        if ok and not _detect_jump(best, sol):
            return sol, True

    # Perturbed guesses
    for scale in [0.01, 0.05, 0.1, 0.2]:
        for _ in range(5):
            guess = prev + np.random.uniform(-scale, scale, size=len(prev))
            sol, ok = _solve_position(solver, theta, params, guess)
            if ok and not _detect_jump(prev, sol):
                return sol, True
            if ok:
                best = sol

    return best, False


# ═══════════════════════════════════════════════════════════════════════
# Grashof pre-feasibility checks
# ═══════════════════════════════════════════════════════════════════════


def _polygon_inequality(lengths):
    """True if the longest side is shorter than the sum of the others."""
    lengths = np.array(lengths)
    return float(lengths.max()) < float(lengths.sum() - lengths.max())


def _grashof_condition(lengths_4):
    """Grashof condition for a 4-link loop: s + l <= p + q."""
    s = sorted(lengths_4)
    return s[0] + s[3] <= s[1] + s[2]


def _check_loop_feasibility(G, link_lengths):
    """Check polygon inequality for every independent loop.

    Parameters
    ----------
    G : nx.Graph
        Mechanism topology.
    link_lengths : list[float]
        Lengths indexed by node.

    Returns
    -------
    bool
        True if all loops satisfy the polygon inequality (necessary for
        the mechanism to close).
    """
    loops = nx.cycle_basis(G)
    for loop in loops:
        loop_lengths = [link_lengths[node] for node in loop]
        if not _polygon_inequality(loop_lengths):
            return False
    return True


# ═══════════════════════════════════════════════════════════════════════
# Joint position computation (for visualization / animation)
# ═══════════════════════════════════════════════════════════════════════


def _edge_key(u, v):
    return (min(u, v), max(u, v))


def _compute_joint_positions(G, angles, link_lengths, input_link):
    """Compute (x, y) of every joint via BFS from the ground link.

    Parameters
    ----------
    G : nx.Graph
    angles : dict[int, float]
        Angle of each link (node 0 = 0).
    link_lengths : list[float]
    input_link : int

    Returns
    -------
    dict[tuple[int,int], ndarray]
        Maps canonical edge ``(min, max)`` to ``[x, y]``.
    """
    ll = link_lengths
    positions = {}

    ground_neighbors = sorted(G.neighbors(0))
    l0 = ll[0]

    # Ground pivots along x-axis
    positions[_edge_key(0, input_link)] = np.array([0.0, 0.0])
    others = [n for n in ground_neighbors if n != input_link]
    for i, gn in enumerate(others):
        frac = 1.0 if len(others) == 1 else (i + 1) / len(others)
        positions[_edge_key(0, gn)] = np.array([l0 * frac, 0.0])

    visited = {0}
    queue = list(ground_neighbors)
    for _ in range(5 * G.number_of_nodes()):
        if not queue:
            break
        link = queue.pop(0)
        if link in visited:
            continue

        known_pos, known_nb = None, None
        for nb in sorted(G.neighbors(link)):
            key = _edge_key(link, nb)
            if key in positions:
                known_pos, known_nb = positions[key], nb
                break
        if known_pos is None:
            queue.append(link)
            continue

        visited.add(link)
        theta = angles.get(link, 0.0)
        l = ll[link]
        d = np.array([np.cos(theta), np.sin(theta)])
        perp = np.array([-np.sin(theta), np.cos(theta)])

        unplaced = [
            nb
            for nb in sorted(G.neighbors(link))
            if _edge_key(link, nb) not in positions
        ]
        if len(unplaced) == 1:
            positions[_edge_key(link, unplaced[0])] = known_pos + l * d
        elif len(unplaced) >= 2:
            positions[_edge_key(link, unplaced[0])] = known_pos + l * d
            for j, nb in enumerate(unplaced[1:], 1):
                positions[_edge_key(link, nb)] = (
                    known_pos + 0.5 * l * d + ((-1) ** j) * 0.35 * l * perp
                )

        for nb in sorted(G.neighbors(link)):
            if nb not in visited:
                queue.append(nb)

    return positions


def _compute_coupler_point(solution, link_lengths, coupler_link, coupler_offset,
                           input_angle, input_link, unknown_indices):
    """Compute the (x, y) position of the coupler point."""
    angles = {0: 0.0, input_link: input_angle}
    for i, idx in enumerate(unknown_indices):
        angles[idx] = solution[i]

    # Trace chain from ground pivot through input link to coupler link
    pos = np.array([0.0, 0.0])
    chain = (
        list(range(input_link, coupler_link + 1))
        if input_link <= coupler_link
        else list(range(input_link, coupler_link - 1, -1))
    )

    for link_idx in chain:
        theta = angles.get(link_idx, 0.0)
        length = link_lengths[link_idx]
        pos = pos + length * np.array([np.cos(theta), np.sin(theta)])
        if link_idx == coupler_link:
            pos_start = pos - length * np.array([np.cos(theta), np.sin(theta)])
            dx, dy = coupler_offset
            c, s = np.cos(theta), np.sin(theta)
            return pos_start + np.array([dx * c - dy * s, dx * s + dy * c])

    theta_c = angles.get(coupler_link, 0.0)
    dx, dy = coupler_offset
    c, s = np.cos(theta_c), np.sin(theta_c)
    return pos + np.array([dx * c - dy * s, dx * s + dy * c])


# ═══════════════════════════════════════════════════════════════════════
# FK result container (compatible with FourBar/FiveBar pattern)
# ═══════════════════════════════════════════════════════════════════════


class FKResult:
    """Result from :meth:`NBarMechanism.FK`, mimicking scipy's result object.

    Attributes
    ----------
    cost : float
        0.0 if converged, otherwise the max absolute residual.
    converged : bool
    singular : bool
    """

    def __init__(self, converged, cost=0.0, singular=False):
        self.cost = cost
        self.converged = converged
        self.singular = singular

    def __repr__(self):
        return f"FKResult(converged={self.converged}, cost={self.cost:.2e})"


# ═══════════════════════════════════════════════════════════════════════
# NBarMechanism class
# ═══════════════════════════════════════════════════════════════════════


class NBarMechanism:
    """General planar N-bar linkage mechanism.

    Automatically derives loop-closure equations from a topology graph
    and compiles fast numerical solvers with analytical Jacobians.

    Parameters
    ----------
    edges_or_graph : list[tuple[int,int]] or nx.Graph
        Topology as an edge list ``[(0,1), (1,2), ...]`` or a NetworkX graph.
        Node 0 is always the ground (fixed) link.
    link_lengths : list[float] or dict[int, float]
        Length of each link, indexed by node number.
    input_link : int, optional
        Node index of the driving (crank) link.  Auto-detected as the first
        binary neighbor of ground if omitted.
    coupler_link : int, optional
        Node index of the link carrying the traced coupler point.
        Auto-detected as the first ternary+ node if omitted.
    coupler_offset : tuple[float, float], optional
        ``(dx, dy)`` offset of the coupler point in the coupler link's local
        frame.  Defaults to ``(0.5 * avg_length, 0.3 * avg_length)``.

    Examples
    --------
    >>> mech = NBarMechanism([(0,1),(1,2),(2,3),(3,0)], [2, 1, 1.5, 1])
    >>> sol, info = mech.FK(0.5)
    >>> info.converged
    True
    """

    def __init__(self, edges_or_graph, link_lengths, input_link=None,
                 coupler_link=None, coupler_offset=None, markers=None):
        # Build graph
        if isinstance(edges_or_graph, nx.Graph):
            self.G = edges_or_graph.copy()
        else:
            self.G = nx.Graph()
            self.G.add_edges_from(edges_or_graph)

        n = self.G.number_of_nodes()
        if isinstance(link_lengths, dict):
            self.lengths = [float(link_lengths.get(i, 1.0)) for i in range(n)]
        else:
            self.lengths = [float(x) for x in link_lengths]

        # Input link auto-detection
        self._input_link = input_link if input_link is not None else _find_input_link(self.G)

        # Markers: list of (link_index, (dx, dy)) in link's local frame
        # If markers provided, use directly. Otherwise build from coupler_link/coupler_offset.
        if markers is not None:
            self._markers = markers
        else:
            if coupler_link is not None:
                self._coupler_link = coupler_link
            else:
                self._coupler_link = self._auto_coupler()
            if coupler_offset is not None:
                self._coupler_offset = tuple(coupler_offset)
            else:
                avg = float(np.mean(self.lengths))
                self._coupler_offset = (0.5 * avg, 0.3 * avg)
            self._markers = [(self._coupler_link, self._coupler_offset)]
        self._marker_bars = [link for link, _ in self._markers]

        # Spatial transform (matching FourBar / FiveBar interface)
        self.oloc = np.array([0.0, 0.0])
        self.rotm = np.eye(2)

        # Derive and compile solver
        self._loop_result = _derive_loop_closure(self.G)
        self._solver = _compile_solver(self._loop_result)
        self._input_link_idx = self._loop_result.input_link
        self._unknown_indices = self._extract_unknown_indices()

        # Warm-start guess
        self.zpos = np.zeros(len(self._solver.unknown_names))

    # ── helpers ────────────────────────────────────────────────────────

    def _auto_coupler(self):
        for node in sorted(self.G.nodes()):
            if node == 0:
                continue
            if self.G.degree(node) >= 3:
                return node
        return min(2, self.G.number_of_nodes() - 1)

    def _extract_unknown_indices(self):
        indices = []
        for name in self._solver.unknown_names:
            m = re.search(r"theta(\d+)", name)
            if m:
                indices.append(int(m.group(1)))
        return indices

    # ── spatial transform (FourBar / FiveBar compatible) ──────────────

    def setOloc(self, x, y):
        """Set the mechanism's origin offset in the global frame."""
        self.oloc = np.array([x, y])

    def setRotm(self, rotm):
        """Set the mechanism's rotation matrix."""
        self.rotm = np.asarray(rotm)

    def setRot(self, theta):
        """Set the mechanism's rotation by angle *theta* (radians)."""
        c, s = np.cos(theta), np.sin(theta)
        self.rotm = np.array([[c, -s], [s, c]])

    # ── forward kinematics ────────────────────────────────────────────

    def FK(self, theta1, zpos=None):
        """Solve forward kinematics at crank angle *theta1*.

        Parameters
        ----------
        theta1 : float
            Input (crank) angle in radians.
        zpos : ndarray, optional
            Initial guess for unknowns.  Uses the last converged solution
            if omitted (continuation method).

        Returns
        -------
        solution : ndarray
            Solved unknown angles.
        info : FKResult
            Result object with ``.cost`` and ``.converged`` attributes,
            compatible with the ``FourBar.FK`` return convention.
        """
        if zpos is None:
            zpos = self.zpos
        sol, ok = _solve_position(self._solver, theta1, self.lengths, zpos)
        if not ok:
            # Try random restarts
            init = _find_initial_solution(
                self._solver, theta1, self.lengths, n_tries=50
            )
            if init is not None:
                sol, ok = init, True
        if ok:
            self.zpos = sol.copy()
            return sol, FKResult(converged=True)
        else:
            param_vals = list(self.lengths) + [theta1]
            res = np.array(
                self._solver.f_compiled(*list(sol), *param_vals), dtype=float
            ).flatten()
            cost = float(np.max(np.abs(res)))
            return sol, FKResult(converged=False, cost=cost)

    # ── joint positions ───────────────────────────────────────────────

    def ComputePoints(self, theta1, solution=None):
        """Compute all joint positions at crank angle *theta1*.

        Parameters
        ----------
        theta1 : float
            Input crank angle.
        solution : ndarray, optional
            Pre-solved unknowns.  If omitted, ``FK`` is called first.

        Returns
        -------
        dict[str, ndarray]
            Joint positions keyed by ``"J_u_v"`` (moving joints),
            ``"G_i"`` (ground pivots), and ``"P"`` (coupler point).
            All positions are in the global frame after applying
            ``rotm`` and ``oloc``.
        """
        if solution is None:
            solution, info = self.FK(theta1)
            if not info.converged:
                # Try bootstrap with random restarts
                init = _find_initial_solution(
                    self._solver, theta1, self.lengths, n_tries=50
                )
                if init is None:
                    return {}
                solution = init
                self.zpos = init.copy()

        # Build angle map
        angles = {0: 0.0, self._input_link_idx: theta1}
        for i, idx in enumerate(self._unknown_indices):
            angles[idx] = solution[i]

        raw = _compute_joint_positions(
            self.G, angles, self.lengths, self._input_link_idx
        )

        # Transform to global frame and assign readable names
        points = {}
        for (u, v), pos in raw.items():
            transformed = self.rotm @ pos + self.oloc
            if u == 0 or v == 0:
                other = v if u == 0 else u
                points[f"G{other}"] = transformed
            else:
                points[f"J{u}_{v}"] = transformed

        # Marker points
        for j, (link, (dx, dy)) in enumerate(self._markers):
            theta = angles.get(link, 0.0)
            # Find base joint (first placed neighbor)
            base = None
            for nb in sorted(self.G.neighbors(link)):
                key = _edge_key(link, nb)
                if key in raw:
                    base = raw[key]
                    break
            if base is not None:
                c, s = np.cos(theta), np.sin(theta)
                mp = base + np.array([dx * c - dy * s, dx * s + dy * c])
                points[f"marker_{j+1}"] = self.rotm @ mp + self.oloc

        return points

    # ── Grashof check ─────────────────────────────────────────────────

    def CheckGrashof(self):
        """Check if all independent loops satisfy necessary closure conditions.

        Returns
        -------
        bool
            True if every loop satisfies the polygon inequality and (for
            4-link loops) the Grashof condition.
        """
        loops = nx.cycle_basis(self.G)
        for loop in loops:
            loop_lengths = [self.lengths[n] for n in loop]
            if not _polygon_inequality(loop_lengths):
                return False
            if len(loop) == 4 and not _grashof_condition(loop_lengths):
                return False
        return True

    # ── plotting ──────────────────────────────────────────────────────

    def plot(self, theta1, ax=None, solution=None):
        """Render the mechanism at crank angle *theta1*.

        Parameters
        ----------
        theta1 : float
        ax : matplotlib.axes.Axes, optional
        solution : ndarray, optional

        Returns
        -------
        matplotlib.axes.Axes
        """
        if ax is None:
            ax = plt.gca()

        points = self.ComputePoints(theta1, solution)
        if not points:
            return ax

        # Rebuild raw edge-keyed positions for link rendering
        if solution is None:
            solution, _ = self.FK(theta1)
        angles = {0: 0.0, self._input_link_idx: theta1}
        for i, idx in enumerate(self._unknown_indices):
            angles[idx] = solution[i]
        raw_pos = _compute_joint_positions(
            self.G, angles, self.lengths, self._input_link_idx
        )

        # Apply transform
        pos = {k: self.rotm @ v + self.oloc for k, v in raw_pos.items()}

        # Ground link
        self._bar_colors = {0: 'k'}
        bar_colors = self._bar_colors
        g_pts = [pos[_edge_key(0, nb)] for nb in sorted(self.G.neighbors(0))
                 if _edge_key(0, nb) in pos]
        if len(g_pts) >= 2:
            ax.plot([g_pts[0][0], g_pts[-1][0]],
                    [g_pts[0][1], g_pts[-1][1]], 'k')

        # Moving links — use consistent color per link
        for node in sorted(self.G.nodes()):
            if node == 0:
                continue
            link_pts = [pos[_edge_key(node, nb)]
                        for nb in sorted(self.G.neighbors(node))
                        if _edge_key(node, nb) in pos]
            color = None
            for i in range(len(link_pts) - 1):
                p0, p1 = link_pts[i], link_pts[i + 1]
                if color is None:
                    line = ax.plot([p0[0], p1[0]], [p0[1], p1[1]])[0]
                    color = line.get_color()
                else:
                    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color)
            if color:
                bar_colors[node] = color

        # Joints
        for (u, v), p in pos.items():
            ax.plot(p[0], p[1], 'ko')

        # Marker lines from link joint centroid
        for j, link in enumerate(self._marker_bars):
            k = f'marker_{j+1}'
            if k not in points:
                continue
            link_joints = [pos[_edge_key(link, nb)]
                           for nb in sorted(self.G.neighbors(link))
                           if _edge_key(link, nb) in pos]
            if link_joints:
                mid = np.mean(link_joints, axis=0)
                c = bar_colors.get(link, 'k')
                ax.plot([mid[0], points[k][0]], [mid[1], points[k][1]], '-', color=c)
                ax.plot(points[k][0], points[k][1], '*', color=c, markersize=10)

        return ax


# ═══════════════════════════════════════════════════════════════════════
# Module-level functions (matching fourbar.py / fivebar.py pattern)
# ═══════════════════════════════════════════════════════════════════════


def GetTrajectory(mech, n_points=360):
    """Sweep the crank through a full revolution and trace marker points.

    Uses continuation (previous solution as next guess) with branch-jump
    detection and recovery for robust multi-loop mechanisms.

    Parameters
    ----------
    mech : NBarMechanism
    n_points : int

    Returns
    -------
    trajectories : dict[str, ndarray]
        ``{'marker_1': array(n_points, 2), ...}`` positions (NaN where solver failed).
    thetas : ndarray, shape (n_points, n_unknowns + 1)
        All angles at each step (column 0 = input angle).
    """
    solver = mech._solver
    params = mech.lengths
    thetas_in = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    n_unknowns = len(solver.unknown_names)
    marker_keys = [f'marker_{j+1}' for j in range(len(mech._markers))]
    trajectories = {k: np.full((n_points, 2), np.nan) for k in marker_keys}
    thetas_out = np.full((n_points, n_unknowns + 1), np.nan)

    # Bootstrap
    init = _find_initial_solution(solver, thetas_in[0], params, n_tries=100)
    if init is None:
        return trajectories, thetas_out

    prev = init.copy()

    for i, theta in enumerate(thetas_in):
        sol, ok = _solve_position(solver, theta, params, prev)
        if not ok:
            alt = _find_initial_solution(solver, theta, params, n_tries=20)
            if alt is not None:
                sol, ok = alt, True

        if ok:
            # Branch-jump handling
            if i > 0 and not np.isnan(thetas_out[i - 1, 0]):
                if _detect_jump(prev, sol):
                    rec, rec_ok = _recover_branch(solver, theta, params, prev)
                    if rec_ok:
                        sol = rec

            thetas_out[i, 0] = theta
            thetas_out[i, 1:] = sol

            points = mech.ComputePoints(theta, sol)
            for k in marker_keys:
                if k in points:
                    trajectories[k][i] = points[k]

            prev = sol.copy()

    return trajectories, thetas_out


def Animate(mech, trajectory_star=None, fps=20):
    """Create a matplotlib animation of the mechanism in motion.

    Shows links, joints, ground, and marker trails.  Optionally
    overlays a target trajectory ``trajectory_star`` for comparison.

    Parameters
    ----------
    mech : NBarMechanism
    trajectory_star : ndarray, shape (M, 2), optional
        Desired trajectory (drawn as blue dots).
    fps : int
        Frames per second (interval = 1000/fps ms).

    Returns
    -------
    ani : matplotlib.animation.FuncAnimation
    fig : matplotlib.figure.Figure
    """
    trajectories, thetas = GetTrajectory(mech)
    marker_keys = list(trajectories.keys())

    # Determine view limits from all marker trajectories + mechanism extent
    all_markers = np.vstack(list(trajectories.values()))
    valid = ~np.isnan(all_markers[:, 0])
    if valid.any():
        all_pts = all_markers[valid]
    else:
        all_pts = np.array([[0, 0], [1, 1]])

    # Also include joint positions from first valid frame
    first_traj = trajectories[marker_keys[0]]
    first_valid_mask = ~np.isnan(first_traj[:, 0])
    first_valid = np.argmax(first_valid_mask) if first_valid_mask.any() else 0
    pts0 = mech.ComputePoints(thetas[first_valid, 0])
    if pts0:
        jpts = np.array([v for k, v in pts0.items() if not k.startswith('marker_')])
        if len(jpts):
            all_pts = np.vstack([all_pts, jpts])

    center = (all_pts.min(axis=0) + all_pts.max(axis=0)) / 2
    span = (all_pts.max(axis=0) - all_pts.min(axis=0)).max()
    span = max(span, 1.0)

    fig, ax = plt.subplots(figsize=(8, 8))

    def update(frame):
        ax.cla()
        theta1 = thetas[frame, 0]
        sol = thetas[frame, 1:]
        if np.isnan(theta1):
            return

        mech.plot(theta1, ax=ax, solution=sol)

        # Marker trails
        for j, k in enumerate(marker_keys):
            c = mech._bar_colors.get(mech._marker_bars[j], 'r')
            traj = trajectories[k]
            trail = traj[:frame + 1]
            trail_valid = ~np.isnan(trail[:, 0])
            if trail_valid.any():
                ax.plot(trail[trail_valid, 0], trail[trail_valid, 1],
                        '-', color=c, lw=2, alpha=0.7)
            if not np.isnan(traj[frame, 0]):
                ax.plot(traj[frame, 0], traj[frame, 1], '*', color=c)

        # Target trajectory
        if trajectory_star is not None:
            ax.plot(trajectory_star[:, 0], trajectory_star[:, 1],
                    "b.", ms=3, alpha=0.5, label="Target")
            ax.legend(fontsize=8, loc="upper right")

        ax.set_xlim(center[0] - span, center[0] + span)
        ax.set_ylim(center[1] - span, center[1] + span)
        ax.set_aspect("equal")

    valid_count = int(valid.sum())
    n_frames = len(thetas) if valid_count > 0 else 1
    ani = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=int(1000 / fps),
    )
    return ani, fig
