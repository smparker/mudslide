# Mudslide 1.0 Release: Suggested Improvements

## Context

Mudslide is a Python nonadiabatic molecular dynamics library at version 0.12.0. Before a 1.0 release, this audit identifies bugs, missing features, inconsistencies, and quality improvements across the entire codebase. Items are grouped by category and prioritized.

---

## 1. Bugs and Correctness Issues

### Critical

- `**qmmm_model.py` — `clone()` fixed**: `clone()` is now correctly defined at class level in `QMMM`.
- `**afssh.py` — AFSSH nstates check fixed**: `gamma_collapse` now raises `NotImplementedError` for >2 states instead of using an `assert`.
- `**surface_hopping_md.py` — bare `except:` fixed**: Now catches `(TypeError, ValueError)` specifically.

### High

- `**afssh.py` — assertions replaced with proper exceptions**: Runtime validation uses `ValueError` and `RuntimeError` instead of `assert` statements. Generic `Exception` raises replaced with `ValueError` and `TypeError`.
- **Version files synced**: `.bumpversion.cfg` updated to 0.12.0, `docs/source/conf.py` now imports version dynamically from `mudslide.version`. Bumpversion config tracks docs/conf.py.

### Medium

- **Energy gap thresholds unified**: Both `DiabaticModel_` and `AdiabaticModel_` now use a `coupling_energy_threshold` class attribute (default `1.0e-14`). Subclasses can override this value.
- `**afssh.py:288` — magic number `1e-10**`: Hardcoded floor for `ddP` to avoid division by zero. Should be a named constant or parameter.
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
- **Parallel trajectory execution** (`batch.py`): Commented-out multiprocessing code (lines 274-299) shows this was planned. For batch runs of hundreds of trajectories, serial execution is a major bottleneck. Implement using `multiprocessing.Pool` or `concurrent.futures.ProcessPoolExecutor`. The `nprocs` option already exists but warns and falls back to serial.

### Medium Priority (would strengthen the package)

- **Trivial crossing / state reordering detection**: Near conical intersections, adiabatic state ordering can swap between timesteps. No detection or handling mechanism exists. At minimum, add a warning; ideally, implement overlap-based state tracking (some infrastructure exists in `DiabaticModel_` phase fixing but not for ab initio models).
- **Generalize AFSSH beyond 2 states**: The collapse equation is hardcoded for 2 states (now raises `NotImplementedError`). The Subotnik group has published multi-state generalizations.

### Lower Priority (nice to have for 1.0)

- **Configuration file support**: All options are CLI arguments or programmatic. A YAML config file would help usability for complex ab initio setups.
- **Adaptive electronic timestep**: The `max_electronic_dt` and `starting_electronic_intervals` options exist but there's no fully adaptive scheme that detects rapid coupling changes and refines automatically.

---

## 3. Feature Inconsistencies

### High

- `**dt` validation in AdiabaticMD**: `AdiabaticMD` now raises a clear `ValueError("dt option is required for AdiabaticMD")` if `dt` is omitted. `SurfaceHoppingMD` still defaults to `fs_to_au`.
- **Ehrenfest `outcome_type` defaulted to `"populations"**`: Ehrenfest now defaults `outcome_type` to `"populations"` since the "state" outcome is meaningless for mean-field dynamics. Docstring documents that `forced_hop_threshold` is inherited but unused.
- `**forced_hop_threshold` accepted by Ehrenfest**: Inherited but is a complete no-op since `surface_hopping()` returns immediately. Should not be in Ehrenfest's `recognized_options`.

### Medium

- **AdiabaticMD has no shared base with SurfaceHoppingMD**: These two classes have significant code duplication (duration handling, snapshot logic, option parsing, propagation loop) but no common base class. A shared `TrajectoryBase` would reduce duplication and ensure feature parity.
- `**needed_gradients()` / `needed_couplings()` convention documented**: Docstrings in `SurfaceHoppingMD`, `Ehrenfest`, `AugmentedFSSH`, and `ElectronicModel_.compute()` now document that returning `None` means "all needed".
- **Two CLI entry points**: `mudslide` (**main**.py) and `mud` (mud.py) overlap in functionality. For 1.0, consolidate into one or clearly document that `mud` is the successor and deprecate `mudslide`.

---

## 4. Code Quality and Static Analysis

### High

- **Pylint configured and cleaned up**: Unused imports removed across 6 files (`Dict`, `Tuple`, `sys`, `math`, `Union`, `numpy`). Codebase formatted with yapf. `.pylintrc` updated for pylint 4.x compatibility (removed deprecated `suggestion-mode`). Score is 9.36/10 with remaining issues being duplicated code, missing docstrings in turboparse, and minor style suggestions. Disabled `too-many-positional-arguments` (R0917) globally as inappropriate for scientific constructors. Disabled `too-many-instance-attributes` (R0902) on `SurfaceHoppingMD` (36 attrs, all necessary). Configured `good-names-rgxs` with 7 regex patterns for scientific naming conventions (Hamiltonians, derivatives, matrix elements, etc.) — all 103 naming violations resolved without disabling the check.
- **CI doesn't run mypy or pylint**: Both are configured (pyproject.toml, .pylintrc) but the GitHub Actions workflow only runs `pytest`. Add mypy and pylint steps to CI. This is especially important for 1.0 to catch type errors.
- **mypy has 885 errors**: Mostly missing type annotations in `turboparse/`, `util.py`, `mud.py`, `__main__.py`, and `turbo_make_harmonic.py`; NumPy `ArrayLike` mismatches with strict stubs; missing `types-PyYAML` stubs; and implicit `Optional` defaults. Needs significant work before it can run in CI.
- **No test coverage reporting**: Add `pytest-cov` and a coverage badge. Important for understanding what's tested before 1.0.
- **Previously empty test files now have tests**: `test_tracer.py` has 2 tests and `test_boltzmann_velocity.py` has 1 test covering core functionality.

### Medium

- **Tests standardized on pytest**: All test files now use pytest style (plain functions/classes, `assert`, `pytest.raises`, fixtures). No remaining `unittest.TestCase` usage. 336 tests pass.
- **No `conftest.py**`: No shared fixtures. Common setup (model creation, trajectory initialization) is duplicated across test files.
- **Under-tested areas**: AFSSH (6 tests), scattering models (2 tests), QM/MM (1 test), OpenMM (1 test), surface CLI (0 tests), quadrature (1 test). Increase coverage for core algorithms before 1.0.
- **Pickle for output** (`__main__.py:220`): `pickle.dump()` used for saving results. Consider safer alternatives (JSON, YAML which is already used elsewhere, or at minimum document the pickle security caveat).

### Low

- **Commented-out code**: Multiple locations (batch.py multiprocessing, surface_hopping_md.py:394 troubleshooter). Clean up or implement.
- **Update GitHub Actions versions**: `actions/checkout@v3` and `actions/setup-python@v3` should be updated to v4.

---

## 5. Packaging and Distribution

### High

- **Versions synced across all files**: `.bumpversion.cfg` matches `version.py` at 0.12.0. `docs/source/conf.py` imports version dynamically. Bumpversion config updated to track all version files.

### Medium

- **Add MANIFEST.in**: Not present — could cause missing files in source distributions (e.g., example files, test data).
- `**dist/` added to .gitignore**.
- **Add `py.typed` marker**: For PEP 561 compliance, add a `py.typed` file so downstream users get type checking benefits.
- **Consider dropping Python 3.9**: If 1.0 is coming soon, 3.9 reaches EOL October 2025. Supporting 3.10+ simplifies code (e.g., `match` statements, `X | Y` union types).

---

## 6. Documentation

### High

- **README updated for 1.0**: Title changed to "Mudslide", added all models (AdiabaticMD, SubotnikModelW/Z, LinearVibronic, TMModel, OpenMM, QMMM, HarmonicModel), added `mud` CLI, fixed propagator names to "exp"/"linear-rk4", fixed `compute()` signature typo, corrected attribute names to private versions (`_hamiltonian`, `_force`, etc.).
- **Add a CHANGELOG**: Essential for 1.0. Document what changed from the pre-1.0 versions.
- `**None` = "all needed" convention documented**: Docstrings for `needed_gradients()`, `needed_couplings()`, and `compute()` now explain this convention.

### Medium

- **Expand Sphinx docs**: The documentation structure is good but some pages may be thin. Ensure all public API classes have autodoc entries.
- **Add a "Getting Started" tutorial**: A Jupyter notebook or script walkthrough that goes from model → single trajectory → batch simulation → analysis.
- **Document decoherence options** (once implemented): Explain when and why to use each correction.

---

## 7. Usability Improvements

### Medium

- **Progress reporting for batch runs**: Long-running batch simulations produce no output. Add a progress bar (e.g., `tqdm`) or periodic log messages showing completed/total trajectories.
- **Better error messages for missing `dt**`: `AdiabaticMD` now fails immediately with a clear `ValueError` if `dt` is not provided.
- **Option validation at init**: Move all option validation to `__init__` rather than failing mid-simulation. The `check_options` utility exists but isn't consistently applied.

### Low

- **Type the `model` parameter**: Throughout the codebase, `model: Any` is used. Use `'ElectronicModel_'` (forward reference) for better IDE support and type checking.
- **Unify exception hierarchy**: Consider a `MudslideError` base exception in `exceptions.py` (which already exists) and derive specific errors from it.

---

## Summary: Suggested Priority Order for 1.0

**Do first (blockers for a credible 1.0):**

1. [x] Fix the 3 bugs (qmmm clone, bare except, AFSSH assert)
2. Add simple decoherence correction (EDC at minimum)
3. [x] Sync versions, fix README inaccuracies
4. Add CHANGELOG
5. [x] Clean up pylint (done: score 9.36/10, naming conventions configured, unused imports removed)
6. Fix mypy errors and add both to CI

**Do next (significantly improves the package):**
7. AIMD → NAMD initialization pathway
8. Implement parallel batch execution
9. [x] Fix feature inconsistencies (dt defaults, Ehrenfest options)
10. Increase test coverage for under-tested areas
11. [x] Document conventions and update Sphinx docs

**If time permits:**
12. NHC thermostat for surface hopping
13. Generalize AFSSH beyond 2 states
14. State reordering detection
15. [x] Standardize tests on pytest

**Also completed:**

- Fix inconsistent energy gap thresholds (unified as class attribute)
- Add `dist/` to .gitignore

