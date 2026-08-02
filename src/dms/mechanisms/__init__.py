"""Planar linkage mechanisms.

- ``FourBar`` / ``FiveBar``: hand-written symbolic classes with closed-form
  position solve and ``gradient()`` via the implicit function theorem.
- ``NBarMechanism``: general single-DOF planar linkage for any topology
  graph — loop-closure derived from ``nx.cycle_basis`` and compiled to
  fast numerical solvers with analytical Jacobians.
- ``TopologyAtlas``: non-isomorphic 6-bar (Watt, Stephenson), 8-bar (16),
  and 10-bar (~230) topologies for plugging into ``NBarMechanism``.
- ``grashof``: per-loop feasibility checks (polygon inequality + Grashof
  condition) as a fast pre-filter before full-revolution sweeps.
"""

from . import grashof
from .atlas import TopologyAtlas
from .fivebar import FiveBar
from .fourbar import FourBar
from .nbar import NBarMechanism

__all__ = [
    "FourBar",
    "FiveBar",
    "NBarMechanism",
    "TopologyAtlas",
    "grashof",
]
