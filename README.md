# Fewest Switches Surface Hopping
Python implementation of Tully's Fewest Switches Surface Hopping (FSSH) for model problems including
a propagator and an implementation of Tully's model problems described in Tully, J.C. _J. Chem. Phys._ (1990) **93** 1061.
The current implementation is limited to one-dimensional potentials with two electronic states.

## Contents
* fssh.py --- FSSH implementation. The FSSH class within fssh.py should be used to run sets of scattering simulations using FSSH.
* tullymodels.py --- Implementation of three models described in Tully's 1990 JCP: TullySimpleAvoidedCrossing, TullyDualAvoidedCrossing,
                      TullyExtendedCouplingReflection.
* surfaces.py --- script to print potential energy surfaces of the models

## Requirements
* numpy
* scipy

## FSSH
Sets of simulations are run using the FSSH class. A FSSH object must be instantiated by passing a model object. All
other options are passed as keyword arguments to the constructor. For example:

    import fssh
    import tullymodels as models

    simple_model = models.SimpleAvoidedCrossing()
    simulator = fssh.FSSH(simple_model, position = -5.0, momentum = 2.0)
    results = simulator.compute()

    print "Probability of reflection on the ground state: %12.f" % results[0]
    print "Probability of transmission on the ground state: %12.f" % results[1]
    print "Probability of reflection on the excited state: %12.f" % results[2]
    print "Probability of transmission on the excited state: %12.f" % results[3]

will run a series of scattering simulations in parallel with a particle starting at -5.0 a.u. and travelling with an initial momentum of 2.0 a.u.

#### Options
* `initial_state` - specify how the initial electronic state is chosen
    * "ground" (default) - start on the ground state
* `mass` - particle mass (default: 2000 a.u.)
* `position` - initial position (default: -5.0 a.u.)
* `momentum` - initial momentum (default: 2.0 a.u.)
* `initial_time` - starting value of `time` variable (default: 0.0 a.u.)
* `dt` - timestep (default: abs(0.05 / velocity)))
* `total_time` - total simulation length (default: 2 * abs(position/velocity))
* `samples` - number of trajectories to run (default: 2000)
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
* `dim(self)` --- returns the number of states in the model (only 2 is supported)

In all cases, the input `x` ought to be a numpy array with 1 element.

The file tullymodels.py implements the three models in Tully's original paper. They are:

* TullySimpleAvoidedCrossing
* TullyDualAvoidedCrossing
* TullyExtendedCouplingReflection

## Surfaces
Scans of the surfaces can be printed using the surfaces.py script. For usage, call

    python surfaces.py -h

# Notes
* In it's current form, FSSH will definitely break for more than two electronic states.
* This package is written to take advantage of doxygen!
