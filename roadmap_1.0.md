# Mudslide 1.0 Release: Suggested Improvements

## Context

Mudslide is a Python nonadiabatic molecular dynamics library at version 0.12.0. Before a 1.0 release, this audit identifies bugs, missing features, inconsistencies, and quality improvements across the entire codebase. Items are grouped by category and prioritized.

---

## 1. Bugs and Correctness Issues

### Medium

- `**afssh.py:288` — magic number `1e-10`**: Hardcoded floor for `ddP` to avoid division by zero. Should be a named constant or parameter.
- **Multiple hardcoded tolerances**: `1e-8` in time comparisons, `1e-10` placeholders throughout. Define named constants (e.g., `COUPLING_THRESHOLD`, `TIME_TOLERANCE`).

---

## 2. Missing Features

### High Priority (expected for a serious NAMD code)

- **Simple decoherence corrections for FSSH** (`surface_hopping_md.py`): Standard FSSH has *no* decoherence correction. The literature considers this a known deficiency. At minimum, add:
  - **Energy-based decoherence (EDC)** from Granucci & Persico, *J. Chem. Phys.* 2007 — simple, widely used, only a few lines of code
  - Optionally: Simplified Decay of Mixing (SDM, Truhlar) or Instantaneous Decoherence Correction (IDC)
  - These should be opt-in via a `decoherence` keyword option.
- **AIMD → NAMD initialization pathway**: Currently no built-in way to take snapshots from an adiabatic MD (ground-state) trajectory and use them to initialize surface hopping trajectories. This is a very common workflow (run AIMD to equilibrate, then spawn NAMD from snapshots). Needs:
  - A utility/trajectory generator that reads AIMD trace output (positions, velocities) and spawns FSSH initial conditions from them
  - Could be a new `TrajGen` subclass like `TrajGenFromTrace` or `TrajGenFromAIMD`

### Medium Priority (would strengthen the package)

- **Trivial crossing / state reordering detection**: Near conical intersections, adiabatic state ordering can swap between timesteps. No detection or handling mechanism exists. At minimum, add a warning; ideally, implement overlap-based state tracking (some infrastructure exists in `DiabaticModel_` phase fixing but not for ab initio models).
- **Generalize AFSSH beyond 2 states**: The collapse equation is hardcoded for 2 states (now raises `NotImplementedError`). The Subotnik group has published multi-state generalizations.

### Lower Priority (nice to have for 1.0)

- **Configuration file support**: All options are CLI arguments or programmatic. A YAML config file would help usability for complex ab initio setups.

---

## 3. Feature Inconsistencies

### High

- `**forced_hop_threshold` accepted by Ehrenfest**: Inherited but is a complete no-op since `surface_hopping()` returns immediately. Should not be in Ehrenfest's `recognized_options`.

### Medium

- **Two CLI entry points**: `mudslide` (**main**.py) and `mud` (mud.py) overlap in functionality. For 1.0, consolidate into one or clearly document that `mud` is the successor and deprecate `mudslide`.

---

## 4. Code Quality and Static Analysis

### Medium

- **No `conftest.py`**: No shared fixtures. Common setup (model creation, trajectory initialization) is duplicated across test files.
- **Under-tested areas**: AFSSH (6 tests), scattering models (2 tests), QM/MM (1 test), OpenMM (1 test), surface CLI (0 tests), quadrature (1 test). Increase coverage for core algorithms before 1.0.
- **Pickle for output** (`__main__.py:220`): `pickle.dump()` used for saving results. Consider safer alternatives (JSON, YAML which is already used elsewhere, or at minimum document the pickle security caveat).

### Low

- **Commented-out code**: surface_hopping_md.py:394 troubleshooter. Clean up or implement.

---

## 5. Packaging and Distribution

### Medium

- **Add MANIFEST.in**: Not present — could cause missing files in source distributions (e.g., example files, test data).
- **Add `py.typed` marker**: For PEP 561 compliance, add a `py.typed` file so downstream users get type checking benefits.
- **Consider dropping Python 3.9**: If 1.0 is coming soon, 3.9 reaches EOL October 2025. Supporting 3.10+ simplifies code (e.g., `match` statements, `X | Y` union types).

---

## 6. Documentation

### High

- **Add a CHANGELOG**: Essential for 1.0. Document what changed from the pre-1.0 versions.

### Medium

- **Expand Sphinx docs**: The documentation structure is good but some pages may be thin. Ensure all public API classes have autodoc entries.
- **Add a "Getting Started" tutorial**: A Jupyter notebook or script walkthrough that goes from model → single trajectory → batch simulation → analysis.
- **Document decoherence options** (once implemented): Explain when and why to use each correction.

---

## 7. Usability Improvements

### Medium

- **Progress reporting for batch runs**: Long-running batch simulations produce no output. Add a progress bar (e.g., `tqdm`) or periodic log messages showing completed/total trajectories.
- **Option validation at init**: Move all option validation to `__init__` rather than failing mid-simulation. The `check_options` utility exists but isn't consistently applied.

### Low

- **Type the `model` parameter**: Throughout the codebase, `model: Any` is used. Use `'ElectronicModel_'` (forward reference) for better IDE support and type checking.
- **Unify exception hierarchy**: Consider a `MudslideError` base exception in `exceptions.py` (which already exists) and derive specific errors from it.

---

## Summary: Suggested Priority Order for 1.0

**Do first (blockers for a credible 1.0):**

1. Add simple decoherence correction (EDC at minimum)
2. Add CHANGELOG

**Do next (significantly improves the package):**

3. AIMD → NAMD initialization pathway
4. Increase test coverage for under-tested areas

**If time permits:**

5. Generalize AFSSH beyond 2 states
6. State reordering detection
