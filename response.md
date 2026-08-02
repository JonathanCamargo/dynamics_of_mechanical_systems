# Response to GraphThe: `dms_extend` Absorbed

**To:** GraphThe maintainers
**From:** dms maintainers
**Re:** `TODO.md` — *Proposal to dms: Absorb graphthe.dms_extend into the Library*

Thanks for the detailed writeup. We read through `dms_extend/` module by
module and compared against the current state of dms. Short version: we
took what was additive, left what was duplicative, and the pieces we took
are now on `main`. Details, deltas, and what we need back from you below.

---

## Up Front: The Framing Was Out of Date

> "dms currently stops at 5-bar. `FourBar` and `FiveBar` are hand-written
> classes; there's no path to 6/7/8/9/10-bar analysis."

Was true at some point. Not today. The repo already had
`src/dms/mechanisms/nbar.py` (~935 LOC, commit `6c8bd4b` "included markers
and nbar class") with a `NBarMechanism` class that:

- derives loop-closure equations from any topology via `nx.cycle_basis`
  (parallels your `loop_closure.py`);
- compiles `lambdify(cse=True)` solvers with analytical Jacobians
  (parallels `position_analysis.py`);
- handles branch-jump detection and recovery during full-revolution sweeps
  (parallels `singularity.py` — `_detect_jump` / `_recover_branch`);
- ships `GetTrajectory` and `Animate` at module level matching the
  `fourbar.py` / `fivebar.py` pattern;
- supports **multiple markers** per mechanism (`markers=[(link, (dx,dy)), ...]`),
  which `dms_extend` does not;
- has an `nbar.ipynb` example at `examples/mechanisms/` already in use.

So the headline ask — "add an `NBarLinkage` class" — was already done.
You can consume `dms.mechanisms.NBarMechanism` immediately; no migration
needed on that front. Most of `dms_extend/` was parallel work to what
already shipped.

---

## What Landed on `main`

`dms_extend/` has been removed. Two modules were absorbed verbatim (with
minor polish) and `nbar.py` was refactored to delegate to the new
`grashof` module instead of carrying its own inline helpers.

### `dms.mechanisms.atlas` — **new** (from `topology.py`)

Full port of the 6/8-bar hardcoded topologies (Watt, Stephenson, all 16
8-bar class topologies) and the Baranov-filtered 10-bar enumeration
matching the published count of 230.

One small addition: `TopologyAtlas(n_links=8, validate=False)` — an
escape hatch for anyone probing novel atlases where the hard-raises on
count/class distribution would get in the way. Default stays `True`.

Smoke-tested in this commit:

```python
from dms.mechanisms import TopologyAtlas, NBarMechanism
a6 = TopologyAtlas(n_links=6)         # 2 (Watt + Stephenson)
a8 = TopologyAtlas(n_links=8)         # 16 across (4,4,0,0), (5,2,1,0), (6,0,2,0)
mech = NBarMechanism(a6.get_topology('T6B_W'), [1,1,1,2,2,2])  # works end-to-end
```

### `dms.mechanisms.grashof` — **new** (from `grashof.py`)

Ported verbatim with the Ting (1989) references intact. Three public
entry points: `polygon_inequality(lengths)`,
`grashof_condition(lengths_4)`, `check_loop_feasibility(loops, lengths)`.

### `dms.mechanisms.nbar` — **refactored**

Private `_polygon_inequality`, `_grashof_condition`, and
`_check_loop_feasibility` helpers were removed in favor of delegating to
`dms.mechanisms.grashof`. The module-level private names are kept as
aliases for backward compatibility with any external code that reached
in, but there's one source of truth now.

`NBarMechanism.CheckGrashof()` collapses to a single delegated call.
Behavior is unchanged; it was asserted against `[2.0, 1.0, 1.5, 1.0]`
before and after the refactor.

### `dms.mechanisms.__init__` — **now populated**

Was an empty file. Now re-exports:

```python
from dms.mechanisms import FourBar, FiveBar, NBarMechanism, TopologyAtlas, grashof
```

---

## What We Didn't Take, and Why

| `dms_extend/` module | Verdict | Reason |
|---|---|---|
| `topology.py` | ✅ absorbed → `dms.mechanisms.atlas` | Filled a genuine gap. Biggest single win. |
| `grashof.py` | ✅ absorbed → `dms.mechanisms.grashof` | Cleaner than the inline helpers. |
| `loop_closure.py` | ❌ skipped | Duplicates `_derive_loop_closure` in `nbar.py`. |
| `position_analysis.py` | ❌ skipped | Duplicates `_compile_solver` / `_solve_position` in `nbar.py`. |
| `singularity.py` | ❌ skipped | Duplicates `_detect_jump` / `_recover_branch` in `nbar.py`. |
| `coupler_curve.py` | ❌ skipped | Superseded by `GetTrajectory` in `nbar.py`, and its `_find_chain` hardcodes sequential indexing (`list(range(input, coupler+1))`) that breaks on topologies where input and coupler aren't sequentially numbered. `GetTrajectory` does BFS from ground. |

The standalone-function-vs-class-method shape was the deciding factor on
the four skipped modules. `nbar.py` is class-oriented and already does
the same work; having a parallel functional API alongside would be
confusing, not additive.

### What's genuinely useful in the skipped modules

Your explicit dataclasses (`LoopClosureResult`, `CompiledSolver`,
`SingularityInfo`) are cleaner than the private `_Dataclass`-suffixed
versions in `nbar.py`. If you want to open a PR that replaces the
private names in `nbar.py` with those dataclasses as return types, we'd
take it — no public API change, just a testability / readability win.

---

## Real Gaps on Our Side We'd Take Help With

Two things `nbar.py` is missing that `dms_extend/` doesn't solve either
but that GraphThe will trip over as soon as it depends on dms:

1. **`NBarMechanism.gradient()` via implicit function theorem.**
   `FourBar.gradient()` and `FiveBar.gradient()` both compute
   `dP/dl = dP/dθ · (dF/dθ)⁻¹ · (-dF/dl) + dP/dl_direct`. `NBarMechanism`
   has the compiled Jacobians sitting there (`self._solver.J_compiled`)
   but no user-facing gradient method. This is a prerequisite for any
   gradient-based synthesis pipeline on 6-bar and above. Probably ~50
   LOC.

2. **Exporter integration.** `dms.exporter.ExportFunPy` is stable —
   answers your open question #4. Wiring it into `NBarMechanism` so a
   downstream user can ship a solver without the `sympy` import at
   runtime is ~30 LOC.

If you want to contribute (1) as part of your post-paper cleanup, that
would land cleanly on top of what's already there. It's the same
generalization your 10-bar synthesis pipeline needs internally anyway.

---

## Tests

`dms_extend/` didn't ship tests in this tree, so nothing to port
automatically. When you send over your test suite, please retarget:

- `test_topology.py` → `tests/test_atlas.py`, pointed at
  `dms.mechanisms.atlas.TopologyAtlas`. Atlas counts, isomorphism
  uniqueness, and Baranov validity are exactly what we want in CI.
- `test_loop_closure.py` / `test_position_analysis.py` /
  `test_singularity.py` → write as black-box tests against
  `NBarMechanism.FK`, `.ComputePoints`, `.CheckGrashof`, and
  `GetTrajectory`, not the private `_`-prefixed helpers. That way the
  tests survive an internal refactor.
- `test_coupler_curve.py` → retarget at `GetTrajectory(mech)`.

pytest is already in `requirements.txt`. If your tests need
`hypothesis`, we'll add it to `[project.optional-dependencies].dev` when
the PR comes in.

---

## Dependencies

`pyproject.toml` already lists `networkx`, `scipy`, `sympy`, `numpy`,
`matplotlib`. No changes needed. Close that bullet.

---

## Your Open Questions — Resolved

1. **Naming.** `dms.mechanisms.NBarMechanism` (not `NBarLinkage`) —
   already decided by the pre-existing code, and consistent with
   `dms.mechanisms.FourBar` / `FiveBar`. Re-exports live in
   `dms.mechanisms.__init__`.
2. **Atlas storage.** Embedded constants for 6/8-bar (tiny). The 10-bar
   path still enumerates at construction — fine for the paper
   timeline. Externalizing to `dms/data/atlases/10bar.json` so
   `TopologyAtlas(n_links=10)` is deterministic and sub-second would be
   a welcome follow-up PR; happy to take it whenever you get to it.
3. **`ReferenceFrame` integration.** Keep cos/sin. `NBarMechanism`
   already uses it and benchmarks favorably; `FourBar` / `FiveBar` keep
   `ReferenceFrame` because their `gradient()` path leans on it. Not a
   consistency issue.
4. **Exporter reuse.** `dms.exporter.ExportFunPy` is stable — depend on
   it. See gap #2 above.
5. **Gear-ratio extension.** Stays `FiveBar`-only. No generalized
   `constraint_couplings` in `NBarMechanism`. Revisit if a concrete
   second use case shows up.
6. **Version pinning.** We'll cut a tagged release once gradient +
   exporter land (or before paper submission, whichever comes first) so
   GraphThe can pin `dynamics_of_mechanical_systems>=X.Y`. We'll aim to
   preserve the API through at least one minor version after that.

---

## What's Next

No PR needed from your side for the upstreaming itself — it's done. Two
things to send over when convenient:

1. **Your tests.** Retargeted as above. These are the biggest remaining
   gap in our CI, particularly for the atlases.
2. **Optional but welcome: `NBarMechanism.gradient()`**, and the
   `export()` / `load()` round-trip from your proposal §Proposed
   NBarLinkage API. Both are small and neither depends on the other.

If the timing around your paper submission tightens, ping us — we can
cut the tag on short notice.

— dms maintainers
