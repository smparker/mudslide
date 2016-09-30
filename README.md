# Fewest Switches Surface Hopping
Python implementation of Tully's Fewest Switches Surface Hopping (FSSH) for model problems including
a propagator and an implementation of Tully's model problems described in Tully, J.C. _J. Chem. Phys._ (1990) **93** 1061.
The current implementation probably works for more than two electronic states, is completely untested for more than
one dimensional potentials.

## Contents
* fssh.py --- FSSH implementation. The FSSH class within fssh.py should be used to run sets of scattering simulations using FSSH.
* tullymodels.py --- Implementation of several models (three in Tully's 1990 JCP, and a super exchange model from Prezhdo):
  - TullySimpleAvoidedCrossing
  - TullyDualAvoidedCrossing
  - TullyExtendedCouplingReflection
  - SuperExchange
* surfaces.py --- script to print potential energy surfaces of the models

## Requirements
* numpy

## FSSH
Sets of simulations are run using the `BatchedTraj` class. A `BatchedTraj` object must be instantiated by passing a model object
(handles electronic PESs and couplings), a check_end class whose objects decide when to quit the simulation, and a traj_gen
generator that generators new initial conditions. Some simple canned examples are provided for check_end and traj_gen. All
other options are passed as keyword arguments to the constructor. The `compute()` function of the 
`BatchedTraj` object returns a `TraceManager` object that contains all the results. Custom `TraceManager`s can also be
provided. For example:

    import fssh
    import tullymodels as models

    simple_model = models.TullySimpleAvoidedCrossing()

    fssh.CheckEnd.box_bounds = 4 # quit the simulation once the particle leaves the box [-4,4]
    fssh.CheckEnd.nsteps = 10000 # quit the simulation after 10000 time steps

    # Generates trajectories always with starting position -5, starting momentum 2, on ground state
    traj_gen = fssh.TrajGenConst(-5.0, 9.0, "ground")

    simulator = fssh.BatchedTraj(simple_model, fssh.CheckEnd, traj_gen, samples = 20)
    results = simulator.compute()
    outcomes = results.outcomes

    print "Probability of reflection on the ground state:    %12.4f" % outcomes[0,0]
    print "Probability of transmission on the ground state:  %12.4f" % outcomes[0,1]
    print "Probability of reflection on the excited state:   %12.4f" % outcomes[1,0]
    print "Probability of transmission on the excited state: %12.4f" % outcomes[1,1]

will run 20 scattering simulations in parallel with a particle starting at -5.0 a.u. and travelling with an initial momentum of 2.0 a.u.

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
* `nprocs` - number of processes over which to parallelize trajectories (default: number of CPUs available)
* `outcome_type` - how to count statistics at the end of a trajectory
    * "state" (default) - use the state attribute of the simulation only
    * "populations" - use the diagonals of the density matrix

## Models
The model provided to the FSSH class needs to have three functions implemented:

* `V(self, x)` --- returns (ndim,ndim)-shaped numpy array containing the Hamiltonian matrix at position `x`
* `dV(self, x)` --- returns (ndim,ndim,1)-shaped numpy array containing the gradient of the Hamiltonian matrix at position `x`
* `dim(self)` --- returns the number of states in the model

In all cases, the input `x` ought to be a numpy array with 1 element.

The file tullymodels.py implements the three models in Tully's original paper. They are:

* TullySimpleAvoidedCrossing
* TullyDualAvoidedCrossing
* TullyExtendedCouplingReflection

Oleg Prezhdo's three-state super exchange is also include as `SuperExchange`.

## Trajectory generator and ender
For batch runs, one must tell `BatchedTraj` how to decide on new initial conditions
how to decide when a trajectory has finished. The basic requirements for each of those
is simple.

The structure of these classes is somewhat strange because of the limitations of
multiprocessing in python. To make use of multiprocessing, every object
must be able to be `pickle`d, meaning that multiprocessing inherits all the
same limitations. As a result, when using multiprocessing, the end checking
class and the trajectory generator class must both be fully defined in the default
namespace.

### Checking end of simulation
A class must be provided that will decide when to stop running a simulation. It must
implement a constructor that takes no required arguments and it must implement a
`__call__` function that accepts the current state of the trajectory and returns `True`
if simulation should quit, `False` if it should continue. See the `TrajectorySH` class
to find what information the trajectory object passed to `__call__` has access to.
See canned class `CheckEnd` in fssh.py for an example.

### Generating initial conditions
This should be a generator function that accepts a number of samples
and returns a dictionary with starting conditions filled in, e.g.,
the yield statement should be something like

    yield { "position" : x0, "momentum" : p0, "initial_state" : "ground" }

See `TrajGenConst` for an example.

## Surfaces
Scans of the surfaces can be printed using the surfaces.py script. For usage, call

    python surfaces.py -h

# Notes
* This package is written to take advantage of doxygen!
