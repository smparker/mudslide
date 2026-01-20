# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Note:** This file was auto-generated and may contain errors. Please verify commands and details against the actual codebase.

## Project Overview

Mudslide is a Python library for quantum chemistry simulations implementing Fewest Switches Surface Hopping (FSSH) and other nonadiabatic molecular dynamics methods. It supports diabatic and ab initio models with two or more electronic states and one or more dimensional potentials.

## Common Commands

```bash
# Install in development mode
pip install -e .

# Run full test suite
pytest

# Run a single test file
pytest test/test_math.py

# Type checking
mypy mudslide/

# Linting
pylint mudslide/

# Code formatting (Google style)
yapf --style google -i mudslide/*.py
```

## Code Style

- Max line length: 100 characters
- Max module lines: 1000
- Type hints required (mypy enforces `disallow_untyped_defs=True`)
- Scientific variable names allowed: x, y, z, X, Y, Z, v, V, p, P
- Formatting: YAPF with Google style

## Architecture

### Core Trajectory Classes
- `SurfaceHoppingMD` (`surface_hopping_md.py`) - Standard FSSH implementation
- `EvenSamplingTrajectory` (`even_sampling.py`) - FSSH with phase space sampling
- `AugmentedFSSH` (`afssh.py`) - Augmented FSSH variant
- `Ehrenfest` (`ehrenfest.py`) - Ehrenfest dynamics
- `AdiabaticMD` (`adiabatic_md.py`) - Pure adiabatic dynamics

### Electronic Models (`models/`)
- `ElectronicModel_` (`electronics.py`) - Abstract base class for all models
- `DiabaticModel_` (`electronics.py`) - Helper for diabatic models requiring only `V(x)` and `dV(x)` methods
- `scattering_models.py` - Tully models, SuperExchange, Subotnik models
- `turbomole_model.py`, `qmmm_model.py`, `openmm_model.py` - Ab initio integrations

### Batch Simulation
- `BatchedTraj` (`batch.py`) - Manages collections of trajectories
- `TrajGenConst`, `TrajGenNormal` (`traj_gen.py`) - Initial condition generators
- `TraceManager` (`tracer.py`) - Collects and manages simulation results

### Custom Model Interface
Models must derive from `ElectronicModel_` and implement:
- `compute(X, couplings, gradients, reference)` - Computes energies, forces, derivative couplings
- `nstates()` - Returns number of electronic states
- `ndof()` - Returns number of classical degrees of freedom

After `compute()`, the model must store:
- `self.hamiltonian` - nstates x nstates array
- `self.force` - nstates x ndim array
- `self.derivative_coupling` - nstates x nstates x ndim array

For diabatic models, derive from `DiabaticModel_` and implement `V(x)` and `dV(x)` methods, plus define `nstates_` and `ndim_` class variables.

## CLI Entry Points
- `mudslide` - Main CLI for running trajectories
- `mudslide-surface` - Prints 1D surfaces and couplings
- `mud` - Alternative CLI entry
