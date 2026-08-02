"""Per-loop Grashof feasibility pre-filter for N-bar mechanisms.

Provides fast necessary-condition checks on link lengths BEFORE the
expensive full-revolution sweep, by analyzing each independent loop
in the mechanism's cycle basis.

Accuracy and limitations
------------------------
These checks are **necessary but not sufficient**:

- If a check FAILS → the mechanism is **definitely infeasible**
  (no false negatives).
- If all checks PASS → the mechanism **might** be feasible; the
  full-revolution sweep is still needed to confirm.

The gap between necessary and sufficient arises because:

1. Each loop is checked independently, but loops share links and
   constrain each other. A mechanism can satisfy every loop condition
   individually yet still be infeasible due to inter-loop coupling.

2. The Grashof condition (for 4-link loops) guarantees that *some*
   link in the loop can rotate fully, but not necessarily the input
   crank. In a standalone 4-bar, you can determine which link is the
   crank; in a sub-loop of a larger mechanism, the other loops
   constrain which configurations are reachable.

3. For loops with more than 4 links, only the weaker polygon
   inequality is applied (longest side < sum of others), which
   permits many infeasible configurations.

In practice, this filter rejects a large fraction of infeasible
samples at near-zero cost (array comparisons vs. iterative nonlinear
solves), significantly speeding up data generation.

Theory
------
The per-loop approach follows Ting (1989), "Five-bar Grashof criteria",
which showed that rotatability in multi-loop mechanisms decomposes
into per-loop conditions. Each independent loop imposes constraints
analogous to the 4-bar Grashof condition on the links within it.

References
----------
- Grashof, F. (1883). Theoretische Maschinenlehre.
- Ting, K.-L. (1989). "Five-bar Grashof criteria." ASME J. Mechanisms,
  Transmissions, and Automation in Design, 111(3), 357-361.
- Ting, K.-L. (1994). "Mobility criteria of single-loop N-bar linkages."
  ASME J. Mechanical Design, 116(1), 202-208.
"""

from __future__ import annotations


def polygon_inequality(link_lengths) -> bool:
    """Check if a closed polygon can form with the given side lengths.

    The longest side must be strictly shorter than the sum of all other
    sides. This is the generalized triangle inequality for N-gons.

    This is a **necessary condition** for the loop to close at ANY
    configuration. If it fails, no assignment of angles can produce a
    closed loop.

    Parameters
    ----------
    link_lengths : sequence of float
        Lengths of links forming the loop (>= 3 links).

    Returns
    -------
    bool
        True if the polygon inequality is satisfied.
    """
    max_l = max(link_lengths)
    rest_sum = sum(link_lengths) - max_l
    return max_l < rest_sum


def grashof_condition(link_lengths) -> bool:
    """Check the Grashof condition for a 4-link loop.

    For four link lengths sorted as s <= p <= q <= l:
        s + l <= p + q

    Interpretation:
    - **Satisfied**: at least one link can make a full revolution.
    - **Violated**: no link can rotate fully; all links oscillate.

    For a standalone 4-bar this is both necessary and sufficient for
    full rotatability. For a 4-link sub-loop in a larger mechanism,
    it is necessary but not sufficient (other loops add constraints).

    Parameters
    ----------
    link_lengths : sequence of float
        Exactly 4 link lengths.

    Returns
    -------
    bool
        True if the Grashof condition is satisfied.

    Raises
    ------
    ValueError
        If not exactly 4 link lengths are provided.
    """
    if len(link_lengths) != 4:
        raise ValueError(
            f"Grashof condition requires exactly 4 links, got {len(link_lengths)}"
        )
    s, p, q, l = sorted(link_lengths)
    return (s + l) <= (p + q)


def check_loop_feasibility(loops, link_lengths) -> bool:
    """Check per-loop feasibility conditions for a mechanism.

    Applies the strongest available check to each independent loop:
    - 4-link loops: Grashof condition (tighter)
    - 3-link and 5+ link loops: polygon inequality (weaker)

    If ANY loop fails its check, the mechanism is definitely infeasible
    and there is no need to run the expensive full-revolution sweep.

    Parameters
    ----------
    loops : list[list[int]]
        Independent loops from the mechanism's cycle basis. Each loop is
        a list of node indices (link IDs) forming a closed path.
    link_lengths : sequence of float
        Link lengths indexed by node ID: ``link_lengths[node]`` is the
        length of that link.

    Returns
    -------
    bool
        True if all loops pass their feasibility checks.
        False if any loop definitely prevents full-revolution motion.
    """
    for loop in loops:
        loop_lengths = [link_lengths[node] for node in loop]

        if len(loop_lengths) == 4:
            if not grashof_condition(loop_lengths):
                return False
        else:
            if not polygon_inequality(loop_lengths):
                return False

    return True
