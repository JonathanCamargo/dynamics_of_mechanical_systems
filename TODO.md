# Proposal to `dms`: Absorb `graphthe.dms_extend` into the Library

**Audience:** dms maintainers.
**From:** GraphThe (ML-based mechanism synthesis project, companion to dms).
**Status:** proposal / coordination doc. Companion to GraphThe's internal `TODO.md`
(which plans the GraphThe-side migration).

---

## One-Sentence Ask

GraphThe has written ~1,800 lines of general-purpose N-bar linkage analysis
(`graphthe/dms_extend/`) that fills real gaps in dms. We'd like to upstream it
so dms supports arbitrary single-DOF planar linkages, not just 4-bar and 5-bar.

---

## Why dms Should Care

1. **dms currently stops at 5-bar.** `FourBar` and `FiveBar` are hand-written
   classes; there's no path to 6/7/8/9/10-bar analysis. GraphThe's code
   generalizes the pattern to arbitrary topologies.
2. **GraphThe already uses this code in anger.** It drives a full dataset
   generation + evaluation pipeline for ~230 10-bar topologies. The code is
   battle-tested beyond what a greenfield design would be.
3. **Upstreaming makes GraphThe's publication reproducible in dms.** Reviewers
   at ASME JMD / MMT expect mechanism work to be analyzable in standard tools.
   Without dms absorbing this, the paper ships a PyTorch checkpoint with no
   dms-native way to inspect generated mechanisms.
4. **It's additive, not breaking.** Existing `FourBar` and `FiveBar` behavior
   is preserved — they become thin wrappers over the general class.

---

## What `dms_extend/` Contains

All six modules live at `graphthe/dms_extend/`. Line counts, dependencies, and
responsibilities:

| Module | LOC | Depends on | Responsibility |
|---|---:|---|---|
| `topology.py` | 645 | `networkx` | Enumerate non-isomorphic single-DOF planar linkages via degree-sequence + Baranov filtering. Atlases for 6-bar (2), 8-bar (16), 10-bar (~230). Matches published atlas counts. |
| `loop_closure.py` | 209 | `networkx`, `sympy` | Derive symbolic loop-closure equations for any topology via `nx.cycle_basis`. Returns equations + unknown/parameter symbols in a `LoopClosureResult` dataclass. |
| `position_analysis.py` | 187 | `sympy`, `scipy.optimize` | Compile loop-closure equations to NumPy callables via `lambdify(cse=True)` with analytical Jacobians. Compile-once, solve-many. `fsolve`-based position analysis. |
| `coupler_curve.py` | 401 | the above | Full-revolution sweep with branch-tracking and singularity handling. Returns the traced coupler-point trajectory. |
| `singularity.py` | 214 | `numpy` | Jacobian determinant + condition-number based singularity detection; branch-jump detection across sweep steps. |
| `grashof.py` | 159 | `numpy` | Grashof condition checking (generalized). |
| `__init__.py` | 1 | — | Currently just a docstring: `"Extensions for the dms library, destined for future migration."` |

Total: **1,816 LOC** of production code, already exercised by GraphThe's CI.

---

## Proposed dms Layout After Absorption

```
dms/
├── mechanisms/               # NEW submodule
│   ├── __init__.py           # exports NBarLinkage, TopologyAtlas
│   ├── nbar.py               # NBarLinkage class (from loop_closure + position_analysis + coupler_curve)
│   ├── atlas.py              # TopologyAtlas (from topology.py)
│   ├── singularity.py        # singularity.py (unchanged)
│   └── grashof.py            # grashof.py (unchanged)
├── FourBar.py                # becomes a thin wrapper: NBarLinkage with fixed 4-link topology
├── FiveBar.py                # becomes a thin wrapper: NBarLinkage with 5-link topology + gear ratio
└── ...                       # existing dms modules unchanged
```

No existing dms module is renamed or moved. No public API of dms is broken.

---

## Proposed `NBarLinkage` API

Designed to match the `FourBar` / `FiveBar` pattern so existing dms users feel
at home. Extracted from GraphThe's `TODO.md` proposal, refined here for
upstream fitness.

```python
class NBarLinkage:
    """General single-DOF planar linkage.

    Subsumes FourBar (n=4) and FiveBar (n=5). Supports arbitrary topologies
    enumerated from TopologyAtlas.
    """

    def __init__(
        self,
        topology: nx.Graph,           # nodes = links, edges = joints
        link_lengths: dict[int, float],
        ground_joints: list[tuple[int, int]] | None = None,
        input_link: int = 1,
    ):
        # Sets up symbolic frame, derives loop-closure, compiles solver.

    # dms-style forward kinematics
    def FK(self, theta_input: float, guess: np.ndarray | None = None) -> np.ndarray: ...
    def ComputePoints(self) -> dict[str, np.ndarray]: ...
    def GetTrajectory(self, n_steps: int = 360) -> np.ndarray: ...
    def Animate(self, ax=None, desired_path=None) -> Animation: ...

    # Analysis
    def check_singularity(self, theta: float) -> SingularityInfo: ...
    def is_grashof(self) -> bool: ...
    def coupler_curve(self, coupler_link: int, coupler_point: tuple[float, float]) -> np.ndarray: ...

    # Serialization — uses dms.exporter.ExportFunPy once available
    def export(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> "NBarLinkage": ...
```

### Backwards compatibility for `FourBar` / `FiveBar`

```python
class FourBar(NBarLinkage):
    def __init__(self, l1, l2, l3, l4, **kwargs):
        topology = _standard_fourbar_topology()
        super().__init__(topology, link_lengths={0: l1, 1: l2, 2: l3, 3: l4}, **kwargs)

class FiveBar(NBarLinkage):
    # preserves gear-ratio extension
    ...
```

Every existing `FourBar(...).FK(...)` call continues to work with identical
numerics. GraphThe can provide a test matrix proving bit-equivalence for the
existing 4-bar/5-bar unit tests.

---

## `TopologyAtlas` API

```python
from dms.mechanisms import TopologyAtlas

atlas = TopologyAtlas(n_links=8)   # Returns all 16 non-isomorphic 8-bar topologies
for topo in atlas:                 # topo is an nx.Graph
    mech = NBarLinkage(topo, link_lengths=...)
    curve = mech.GetTrajectory(n_steps=360)
```

Atlas data ships as embedded edge-list constants — no runtime enumeration
required for the n=6, 8, 10 cases. For n>10, `TopologyAtlas(n_links=n, enumerate=True)`
runs the Baranov-filtered enumeration (minutes for n=12).

---

## What GraphThe Already Writes That Belongs in dms

From `TODO.md`, restated for dms context:

| GraphThe file | Migrates to | Keep in GraphThe? |
|---|---|---|
| `graphthe/dms_extend/topology.py` | `dms.mechanisms.atlas` | No — move entirely. |
| `graphthe/dms_extend/loop_closure.py` | internals of `dms.mechanisms.nbar` | No — absorbed into NBarLinkage. |
| `graphthe/dms_extend/position_analysis.py` | internals of `dms.mechanisms.nbar` | No — absorbed. |
| `graphthe/dms_extend/coupler_curve.py` | `dms.mechanisms.nbar` methods | No — absorbed. |
| `graphthe/dms_extend/singularity.py` | `dms.mechanisms.singularity` | No — move. |
| `graphthe/dms_extend/grashof.py` | `dms.mechanisms.grashof` (generalized) | No — move. |
| `graphthe/curve_encoding.py` | stays in GraphThe (delegates to `dms.curves`) | Yes — ML-specific wrapping. |
| `graphthe/canonical.py` | — | Yes — deterministic node ordering is ML-only. |
| `graphthe/graph_encoding.py` | — | Yes — PyG conversion is ML-only. |

The split is clean: anything mechanism-analytical goes to dms; anything
ML-representational stays in GraphThe.

---

## Tests GraphThe Ships With the Migration

GraphThe already has these, covering the code being donated. They can be
ported directly (renamed for dms conventions):

- `tests/test_topology.py` — atlas counts, isomorphism uniqueness, Baranov validity
- `tests/test_loop_closure.py` — equation count and unknown count per topology
- `tests/test_position_analysis.py` — numerical convergence, Jacobian agreement
- `tests/test_coupler_curve.py` — trajectory continuity, closure error
- `tests/test_singularity.py` — singularity detection, branch-jump detection

These are pytest-based and already pass on GraphThe's CI. They assume pytest
and `hypothesis`; if dms uses a different test runner we'd port accordingly.

---

## Dependencies Introduced

dms would acquire two new hard dependencies (both already transitive in
GraphThe):

- `networkx >= 3.0` — for topology graphs and `cycle_basis`
- `scipy >= 1.10` — for `fsolve`; already common in dms workflows

Already in dms (used by the ported code):

- `sympy`, `numpy`, `matplotlib`

---

## Migration Sequence (from dms's perspective)

The corresponding GraphThe-side plan is detailed in `TODO.md` §Phase B-D. The
dms-side work breaks into three reviewable PRs:

1. **PR #1: `dms.mechanisms` submodule + `NBarLinkage` + tests.**
   Additive only. No changes to existing `FourBar`/`FiveBar`. Can merge
   independently and GraphThe can start consuming it immediately.

2. **PR #2: `FourBar` / `FiveBar` reimplemented as `NBarLinkage` subclasses.**
   Keeps the existing public API. Includes numerical-equivalence tests vs.
   the pre-refactor implementations.

3. **PR #3: `TopologyAtlas` + examples.**
   Ships the 6/8/10-bar atlases and adds `examples/8bar.ipynb`,
   `examples/10bar.ipynb`, and `examples/graphthe_integration.ipynb`.

GraphThe can begin the consumer-side migration (`TODO.md` §Phase C) after PR #1
lands.

---

## Open Questions for dms Maintainers

1. **Naming.** Is `dms.mechanisms.NBarLinkage` the right home, or do you prefer
   `dms.NBarLinkage` at top-level alongside `FourBar`/`FiveBar`? Either is fine
   on our side.
2. **Atlas storage format.** Edge lists embedded in Python (current approach),
   or externalized to `dms/data/atlases/*.json`? The 10-bar atlas is ~15 KB.
3. **`ReferenceFrame` integration.** `dms_extend` uses raw cos/sin rather than
   dms's `ReferenceFrame` pattern (a historical choice that sped up compilation).
   Should the upstream version adopt `ReferenceFrame` for consistency, or keep
   the faster cos/sin formulation? We can benchmark both.
4. **Exporter reuse.** `NBarLinkage.export()` should use `dms.exporter.ExportFunPy`
   to serialize compiled solvers — is that API stable enough to depend on?
5. **Gear-ratio extension.** `FiveBar` has a gear-ratio parameter the base
   `NBarLinkage` doesn't need. Do we preserve it as a `FiveBar`-only field or
   generalize it (e.g., `constraint_couplings: dict`)?
6. **Version pinning.** GraphThe currently vendors this code. Post-migration,
   GraphThe will pin a minimum `dms` version. What's the cadence for dms releases?

---

## Contact / Coordination

- GraphThe repo: (local dev; will share URL on PR)
- Primary author on the GraphThe side: Jonathan (jcamargo.co@gmail.com)
- Planned timing: Phase A of `TODO.md` (GraphThe using dms utilities) can start
  immediately. PR #1 (NBarLinkage) targeted **before GraphThe paper submission**
  so the paper can cite dms as the underlying library.
