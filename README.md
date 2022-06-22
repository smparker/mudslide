# Fewest Switches Surface Hopping [![Build Status](https://github.com/smparker/mudslide/actions/workflows/python-package.yml/badge.svg)](https://github.com/smparker/mudslide/actions/workflows/python-package.yml) [![Documentation Status](https://readthedocs.org/projects/mudslide/badge/?version=latest)](https://mudslide.readthedocs.io/en/latest/?badge=latest)
Python implementation of Tully's Fewest Switches Surface Hopping (FSSH) for model problems including
a propagator and an implementation of Tully's model problems described in Tully, J.C. _J. Chem. Phys._ (1990) **93** 1061.
The current implementation works for diabatic as well as ab initio models, with two or more electronic states, and with one or more
dimensional potentials.

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
  - some 2D models
    - `Subotnik2D`
* `mudslide` script that runs simple model trajectories
* `mudslide-surface` script that prints 1D surface and couplings

## Requirements
* numpy
* scipy (for Shin-Metiu model)
* pyyaml
* turboparse

## Setup
Mudslide has switched to a proper python package structure, which means to work properly it now needs to be "installed". The
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
    traj_gen = mudslide.TrajGenConst(-5.0, 10.0, 0)

    simulator = mudslide.BatchedTraj(simple_model, traj_gen, mudslide.TrajectorySH, samples = 4, bounds=[[-4],[4]])
    results = simulator.compute()
    outcomes = results.outcomes

    print("Probability of reflection on the ground state:    %12.4f" % outcomes[0,0])
    print("Probability of transmission on the ground state:  %12.4f" % outcomes[0,1])
    print("Probability of reflection on the excited state:   %12.4f" % outcomes[1,0])
    print("Probability of transmission on the excited state: %12.4f" % outcomes[1,1])

will run 4 scattering simulations with a particle starting in the ground state (`0`) at `x=-5.0` a.u. and traveling with an initial momentum of `10.0` a.u.

#### Options
* `initial_state` - specify the initial electronic state
    * 0 (default) - start on the ground state
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
A primary goal of mudslide is to include the flexibility for users to provide their own
model classes. To interface with mudslide, a model class should derive from `mudslide.ElectronicModel_`
and should implement
* a `compute()` function that computes energies, forces, and derivative couplings
* an `nstates()` function that returns the number of electronic states in the model
* an `ndim()` function that returns the number of classical (vibrational) degrees of freedom in the model
 
### compute() function
The `compute()` function needs to have the following signature:

    def compute(self, X: ArrayLike, couplings: Any = None, gradients: Any None, reference: Any = None) -> None

The `X` input to the `compute()` function is an array of the positions. All other inputs are ignored for now
but will eventually be used to allow the trajectory to enumerate precisely which quantities are desired at
each call.
At the end of the `compute()` function, the object must store
* `self.hamiltonian` - An `nstates x nstates` array of the Hamiltonian (`nstates` is the number of electronic states)
* `self.force` - An `nstates x ndim` array of the force on each PES (`ndim` is the number of classical degrees of freedom)
* `self.derivative_coupling` - An `nstates x nstates x ndim` array where `self.derivative_coupling[i,j,:]` contains <i|d/dR|j>.

See the file `mudslide/turbomole_model.py` for an example of a standalone ab initio model.

### Diabatic Models
For diabatic models, such as Tully's scattering models, there is a helper class, `mudslide.DiabaticModel_` to simplify construction of model classes.

To use it, have your model derive from `mudslide.DiabaticModel_` and then you just need to implement two helper functions:
* `V(self, x)` --- returns (nstates,nstates)-shaped numpy array containing the diabatic Hamiltonian matrix at nuclear position `x`
* `dV(self, x)` --- returns (nstates,nstates,ndim)-shaped numpy array containing the gradient of the diabatic Hamiltonian matrix at nuclear position `x`

And further define two class variables:
* `nstates_` --- the number of electronic states in the model
* `ndim_` --- the number nuclear degrees of freedom in the model

In all cases, the input `x` ought to be a numpy array with `ndim` elements.

See the file `mudslide/scattering_models.py` for examples, which includes, among other models, the original Tully models.
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

    yield { "position" : x0, "momentum" : p0, "initial_state" : 0 }

See `TrajGenConst` for an example.

## Surfaces
Scans of the surfaces can be printed using the `mudslide-surface` command that is included
for the installation. For usage, make sure your `PATH` includes the installation
directory for scripts (e.g., by running
`export PATH=$(python -m site --user-base)/bin:$PATH`) and run

    mudslide-surface -h

