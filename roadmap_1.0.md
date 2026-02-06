# Mudslide 1.0 Release: Suggested Improvements

## Context

Mudslide is a Python nonadiabatic molecular dynamics library at version 0.12.0. Before a 1.0 release, this audit identifies bugs, missing features, inconsistencies, and quality improvements across the entire codebase. Items are grouped by category and prioritized.

---

## 1. Bugs and Correctness Issues

### Critical

- **`qmmm_model.py:131` — `clone()` indented inside `compute()`**: The `clone` method is nested inside `compute()` due to a stray indentation, making it unreachable as a class method. Dedent it to class level.

- **`afssh.py:321` — AFSSH hardcoded to 2 states**: `assert self.model.nstates == 2` in `gamma_collapse`. This is an `assert` (disabled with `python -O`) and silently breaks for >2 states. Should raise a proper `NotImplementedError` at init time, or better yet, generalize the collapse equation.

- **`surface_hopping_md.py:145` — bare `except:`**: Catches `SystemExit` and `KeyboardInterrupt`. Change to `except Exception:` or `except (TypeError, ValueError):`.

### High

- **`afssh.py` — assertions used for runtime validation** (lines 177, 224, 257): Replace all `assert` statements used for input/state validation with proper exceptions (`ValueError`, `RuntimeError`).

- **`afssh.py:187,234` — generic `Exception`**: Should be `ValueError("Unrecognized propagation method: ...")`.

- **Version file mismatch**: `.bumpversion.cfg` says 0.11.0, `version.py` says 0.12.0, `docs/source/conf.py` hardcodes 0.11.0. Sync all three and make docs read version dynamically.

### Medium

- **Inconsistent energy gap thresholds**: `DiabaticModel_` uses `1.0e-10` (electronics.py:462), `AdiabaticModel_` uses `1.0e-14` (electronics.py:651) for derivative coupling denominators. Should either unify or document why they differ. Consider making this a class attribute.

- **`afssh.py:288` — magic number `1e-10`**: Hardcoded floor for `ddP` to avoid division by zero. Should be a named constant or parameter.

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

- **NHC thermostat for surface hopping**: `AdiabaticMD` supports Nose-Hoover chains via `adiabatic_propagator.py`, but FSSH does not. NVT surface hopping is used in condensed-phase applications. The propagator infrastructure is already there — expose it for FSSH.

- **Generalize AFSSH beyond 2 states**: The collapse equation (line 321) is hardcoded for 2 states. The Subotnik group has published multi-state generalizations.

### Lower Priority (nice to have for 1.0)

- **Landau-Zener surface hopping**: A simpler alternative to FSSH, sometimes used as a comparison method. Relatively straightforward to implement.

- **Configuration file support**: All options are CLI arguments or programmatic. A YAML config file would help usability for complex ab initio setups.

- **Adaptive electronic timestep**: The `max_electronic_dt` and `starting_electronic_intervals` options exist but there's no fully adaptive scheme that detects rapid coupling changes and refines automatically.

---

## 3. Feature Inconsistencies

### High

- **`dt` default inconsistency**: `SurfaceHoppingMD` defaults `dt` to `fs_to_au` (~41 a.u. ~ 1 fs). `AdiabaticMD` has *no default* and will crash with an unhelpful error if omitted. Either both should require explicit `dt` (safer for a scientific code — forces users to think about it) or both should have sensible defaults.

- **`outcome_type` accepted by Ehrenfest**: Ehrenfest inherits `outcome_type` from `SurfaceHoppingMD`, but the "state" outcome type is meaningless for Ehrenfest (there's no active state in a physical sense). Should either warn or default to "populations" for Ehrenfest.

- **`forced_hop_threshold` accepted by Ehrenfest**: Inherited but is a complete no-op since `surface_hopping()` returns immediately. Should not be in Ehrenfest's `recognized_options`.

### Medium

- **AdiabaticMD has no shared base with SurfaceHoppingMD**: These two classes have significant code duplication (duration handling, snapshot logic, option parsing, propagation loop) but no common base class. A shared `TrajectoryBase` would reduce duplication and ensure feature parity.

- **`needed_gradients()` / `needed_couplings()` convention undocumented**: Returning `None` means "all needed" — this is a non-obvious convention used throughout the codebase but never documented in docstrings or docs.

- **Two CLI entry points**: `mudslide` (__main__.py) and `mud` (mud.py) overlap in functionality. For 1.0, consolidate into one or clearly document that `mud` is the successor and deprecate `mudslide`.

---

## 4. Code Quality and Static Analysis

### High

- **CI doesn't run mypy or pylint**: Both are configured (pyproject.toml, .pylintrc) but the GitHub Actions workflow only runs `pytest`. Add mypy and pylint steps to CI. This is especially important for 1.0 to catch type errors.

- **No test coverage reporting**: Add `pytest-cov` and a coverage badge. Important for understanding what's tested before 1.0.

- **Empty test files**: `test_tracer.py` and `test_boltzmann_velocity.py` exist but contain 0 test cases. Either fill them or remove them.

### Medium

- **Mixed test frameworks**: 15 files use `unittest.TestCase`, 11 use pytest style. Standardize on pytest for consistency.

- **No `conftest.py`**: No shared fixtures. Common setup (model creation, trajectory initialization) is duplicated across test files.

- **Under-tested areas**: AFSSH (6 tests), scattering models (2 tests), QM/MM (1 test), OpenMM (1 test), surface CLI (0 tests), quadrature (1 test). Increase coverage for core algorithms before 1.0.

- **Pickle for output** (`__main__.py:220`): `pickle.dump()` used for saving results. Consider safer alternatives (JSON, YAML which is already used elsewhere, or at minimum document the pickle security caveat).

### Low

- **Commented-out code**: Multiple locations (batch.py multiprocessing, surface_hopping_md.py:394 troubleshooter). Clean up or implement.

- **Update GitHub Actions versions**: `actions/checkout@v3` and `actions/setup-python@v3` should be updated to v4.

---

## 5. Packaging and Distribution

### High

- **Sync version across all files**: Fix `.bumpversion.cfg` (0.11.0), `docs/source/conf.py` (hardcoded 0.11.0), and ensure `bumpversion` updates docs/conf.py too.

- **Make docs version dynamic**: `docs/source/conf.py` should import version from `mudslide.version` instead of hardcoding.

### Medium

- **Add MANIFEST.in**: Not present — could cause missing files in source distributions (e.g., example files, test data).

- **Add `dist/` to .gitignore**: Currently shows as untracked in git status.

- **Add `py.typed` marker**: For PEP 561 compliance, add a `py.typed` file so downstream users get type checking benefits.

- **Consider dropping Python 3.9**: If 1.0 is coming soon, 3.9 reaches EOL October 2025. Supporting 3.10+ simplifies code (e.g., `match` statements, `X | Y` union types).

---

## 6. Documentation

### High

- **README needs updating for 1.0**:
  - Title says "Fewest Switches Surface Hopping" but the package does much more (Ehrenfest, AFSSH, AIMD, QM/MM)
  - Options section references "exponential" and "ode" propagators but the code uses "exp" and "linear-rk4"
  - Missing mention of `AdiabaticMD`, `mud` CLI, Turbomole/OpenMM integrations
  - `compute()` signature example has a typo: `gradients: Any None` (missing `=`)
  - References `self.hamiltonian` but code uses `self._hamiltonian` (property accessor)

- **Add a CHANGELOG**: Essential for 1.0. Document what changed from the pre-1.0 versions.

- **Document the `None` = "all needed" convention**: For `needed_gradients()`, `needed_couplings()`, document this clearly in base class docstrings and in the custom models guide.

### Medium

- **Expand Sphinx docs**: The documentation structure is good but some pages may be thin. Ensure all public API classes have autodoc entries.

- **Add a "Getting Started" tutorial**: A Jupyter notebook or script walkthrough that goes from model → single trajectory → batch simulation → analysis.

- **Document decoherence options** (once implemented): Explain when and why to use each correction.

---

## 7. Usability Improvements

### Medium

- **Progress reporting for batch runs**: Long-running batch simulations produce no output. Add a progress bar (e.g., `tqdm`) or periodic log messages showing completed/total trajectories.

- **Better error messages**: When required options are missing (like `dt` in AdiabaticMD), fail early with a clear message listing the missing option, not a `KeyError` deep in propagation.

- **Option validation at init**: Move all option validation to `__init__` rather than failing mid-simulation. The `check_options` utility exists but isn't consistently applied.

### Low

- **Type the `model` parameter**: Throughout the codebase, `model: Any` is used. Use `'ElectronicModel_'` (forward reference) for better IDE support and type checking.

- **Unify exception hierarchy**: Consider a `MudslideError` base exception in `exceptions.py` (which already exists) and derive specific errors from it.

---

## Summary: Suggested Priority Order for 1.0

**Do first (blockers for a credible 1.0):**
1. Fix the 3 bugs (qmmm clone, bare except, AFSSH assert)
2. Add simple decoherence correction (EDC at minimum)
3. Sync versions, fix README inaccuracies
4. Add CHANGELOG
5. Add mypy + pylint to CI

**Do next (significantly improves the package):**
6. AIMD → NAMD initialization pathway
7. Implement parallel batch execution
8. Fix feature inconsistencies (dt defaults, Ehrenfest options)
9. Increase test coverage for under-tested areas
10. Document conventions and update Sphinx docs

**If time permits:**
11. NHC thermostat for surface hopping
12. Generalize AFSSH beyond 2 states
13. State reordering detection
14. Standardize tests on pytest
15. Configuration file support
