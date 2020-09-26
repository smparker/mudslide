# Fewest Switches Surface Hopping [![Build Status](https://travis-ci.org/smparker/mudslide.svg?branch=master)](https://travis-ci.org/smparker/mudslide) [![Documentation Status](https://readthedocs.org/projects/mudslide/badge/?version=latest)](https://mudslide.readthedocs.io/en/latest/?badge=latest)[![Coverage Status](https://coveralls.io/repos/github/smparker/mudslide/badge.svg?branch=master)](https://coveralls.io/github/smparker/mudslide?branch=master)
Python implementation of Tully's Fewest Switches Surface Hopping (FSSH) for model problems including
a propagator and an implementation of Tully's model problems described in Tully, J.C. _J. Chem. Phys._ (1990) **93** 1061.
The current implementation probably works for more than two electronic states, is completely untested for more than
one dimensional potentials.

## Contents
* `mudslide` package that contains
  - implementation of all surface hopping methods
    - `TrajectorySH` - FSSH
    - `TrajectoryCum` - FSSH with a cumulative point of view
    - `Ehrenfest` - Ehrenfest dynamics
  - collection of 1D models
    - `TullySimpleAvoidedCrossing`
    - `TullyDualAvoidedCrossing`
    - `TullyExtendedCouplingReflection`
    - `SuperExchange`
    - `SubotnikModelX`
    - `SubotnikModelS`
    - `ShinMetiu`
  - now some 2D models
    - `Subotnik2D`
* `mudslide` script that runs simple model trajectories
* `mudslide-surface` script that prints 1D surface and couplings

## Requirements
* numpy
* scipy (for Shin-Metiu model)

## Setup
FSSH has switched to a proper python package structure, which means to work properly it now needs to be "installed". The
most straightforward way to do this is

    cd /path/to/mudslide
    pip install --user -e .

which install into your user installation dir. You can find out your user installation
directory with the command

    python -m site --user-base

To set up your `PATH` and `PYTHONPATH` to be able to use both the command line scripts
and the python package, use

    export PATH=$(python -m site --user-base)/bin:$PATH
    export PYTHONPATH=$(python -m site --user-base):$PYTHONPATH

## Trajectory Surface Hopping
Sets of simulations are run using the `BatchedTraj` class. A `BatchedTraj` object must be instantiated by passing a model object
(handles electronic PESs and couplings), and a `traj_gen`
generator that generates new initial conditions. Some simple canned examples are provided for `traj_gen`. All
other options are passed as keyword arguments to the constructor. The `compute()` function of the
`BatchedTraj` object returns a `TraceManager` object that contains all the results, but functionally behaves
like a python dictionary. Custom `TraceManager`s can also be
provided. For example:

    import mudslide

    simple_model = mudslide.models.TullySimpleAvoidedCrossing()

    # Generates trajectories always with starting position -5, starting momentum 10.0, on ground state
    traj_gen = mudslide.TrajGenConst(-5.0, 10.0, "ground")

    simulator = mudslide.BatchedTraj(simple_model, traj_gen, mudslide.TrajectorySH, samples = 4)
    results = simulator.compute()
    outcomes = results.outcomes

    print("Probability of reflection on the ground state:    %12.4f" % outcomes[0,0])
    print("Probability of transmission on the ground state:  %12.4f" % outcomes[0,1])
    print("Probability of reflection on the excited state:   %12.4f" % outcomes[1,0])
    print("Probability of transmission on the excited state: %12.4f" % outcomes[1,1])

will run 20 scattering simulations in parallel with a particle starting at -5.0 a.u. and traveling with an initial momentum of 10.0 a.u.

#### Options
* `initial_state` - specify how the initial electronic state is chosen
    * "ground" (default) - start on the ground state
* `mass` - particle mass (default: 2000 a.u.)
* `initial_time` - starting value of `time` variable (default: 0.0 a.u.)
* `dt` - timestep (default: abs(0.05 / velocity)))
* `total_time` - total simulation length (default: 2 * abs(position/velocity))
* `samples` - number of trajectories to run (default: 2000)
* `seed` - random seed for trajectories (defaults however numpy does)
* `propagator` - method used to propagate electronic wavefunction
    * "exponential" (default) - apply exponentiated Hamiltonian via diagonalization
    * "ode" - scipy's ODE integrator
* `nprocs` - number of processes over which to parallelize trajectories (default: 1)
* `outcome_type` - how to count statistics at the end of a trajectory
    * "state" (default) - use the state attribute of the simulation only
    * "populations" - use the diagonals of the density matrix
* `trace_every` - save snapshot data every nth step (i.e., when `nsteps%trace_every==0`)

## Models
The model provided to the FSSH class needs to have three functions implemented:

* `V(self, x)` --- returns (nstates,nstates)-shaped numpy array containing the Hamiltonian matrix at nuclear position `x`
* `dV(self, x)` --- returns (nstates,nstates,ndim)-shaped numpy array containing the gradient of the Hamiltonian matrix at nuclear position `x`
* `nstates(self)` --- returns the number of electronic states in the model
* `ndim(self)` --- returns the number nuclear degrees of freedom in the model

In all cases, the input `x` ought to be a numpy array with `ndim` elements.

For simple diabatic models, there is a helper class, `DiabaticModel_` that will
simplify construction of classes for new models. If you derive your class from
`DiabaticModel_`, you only need to implement `V()` and `dV()` in the diabatic basis,
and define class variables `ndim_` and `nstates_` that hold the number of
nuclear degrees of freedom and the number of electronic states, respectively. See
the file `mudslide/models.py` for examples.

The file models.py implements the three models in Tully's original paper. They are:

* TullySimpleAvoidedCrossing
* TullyDualAvoidedCrossing
* TullyExtendedCouplingReflection

Additional models included are:
* `SuperExchange` - Oleg Prezhdo's three-state super exchange model from Wang, Trivedi, Prezhdo JCTC (2014) doi:10.1021/ct5003835
* `SubotnikModelX` - 'Model X' from Subotnik JPCA (2011) doi: 10.1021/jp206557h
* `SubotnikModelS` - 'Model S' from Subotnik JPCA (2011) doi: 10.1021/jp206557h
* `Subotnik2D` - 2D model from Subotnik JPCA (2011) doi: 10.1021/jp206557h
* `Shin-Metiu`

## Trajectory generator
For batch runs, one must tell `BatchedTraj` how to decide on new initial conditions
and how to decide when a trajectory has finished. The basic requirements for each of those
is simple.

The structure of these classes is somewhat strange because of the limitations of
multiprocessing in python. To make use of multiprocessing, every object
must be able to be `pickle`d, meaning that multiprocessing inherits all the
same limitations. As a result, when using multiprocessing, the trajectory generator class must
be fully defined in the default namespace.

### Generating initial conditions
This should be a generator function that accepts a number of samples
and returns a dictionary with starting conditions filled in, e.g.,
the yield statement should be something like

    yield { "position" : x0, "momentum" : p0, "initial_state" : "ground" }

See `TrajGenConst` for an example.

## Surfaces
Scans of the surfaces can be printed using the `surface` command that is included
for the installation. For usage, make sure your `PATH` includes the installation
directory for scripts (e.g., by running
`export PATH=$(python -m site --user-base)/bin:$PATH`) and run

    surface -h

# Notes
* This package is written to take advantage of doxygen!
