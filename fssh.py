#!/usr/bin/env python

## @package fssh
#  Module responsible for propagating surface hopping trajectories

import numpy as np
import math as m
import multiprocessing as mp
import collections

import sys

## Wrapper around all the information computed for a set of electronics
#  states at a given position: V, dV, eigenvectors, eigenvalues
class ElectronicStates:
    ## Constructor
    # @param V Hamiltonian/potential
    # @param dV Gradient of Hamiltonian/potential
    # @param ref_coeff [optional] set of coefficients (for example, from a previous step) used to keep the sign of eigenvectors consistent
    def __init__(self, V, dV, ref_coeff = None):
        self.V = V
        self.dV = dV
        self.energies, self.coeff = np.linalg.eigh(V)
        if ref_coeff is not None:
            for mo in range(self.nstates()):
                if (np.dot(self.coeff[:,mo], ref_coeff[:,mo]) < 0.0):
                    self.coeff[:,mo] *= -1.0

    ## returns dimension of (electronic) Hamiltonian
    def nstates(self):
        return self.V.shape[1]

    ## returns dimensionality of model (nuclear coordinates)
    def ndim(self):
        return self.dV.shape[0]

    ## returns \f$-\langle \phi_{\mbox{state}} | \nabla H | \phi_{\mbox{state}} \rangle\f$ of Hamiltonian
    # @param state state along which to compute force
    def compute_force(self, state):
        out = np.zeros(self.ndim())
        state_vec = self.coeff[:,state]
        for d in range(self.ndim()):
            out[d] = - (np.dot(state_vec.T, np.dot(self.dV[d,:,:], state_vec)))
        return out

    ## returns \f$\phi_{\mbox{state}} | H | \phi_{\mbox{state}} = \varepsilon_{\mbox{state}}\f$
    def compute_potential(self, state):
        return self.energies[state]

    ## returns \f$\phi_{i} | \nabla_\alpha \phi_{j} = d^\alpha_{ij}\f$
    def compute_derivative_coupling(self, bra_state, ket_state):
        out = np.zeros(self.ndim())
        if (bra_state != ket_state):
            for d in range(self.ndim()):
                out[d] = np.dot(self.coeff[:,bra_state].T, np.dot(self.dV[d,:,:], self.coeff[:,ket_state]))

            dE = self.energies[ket_state] - self.energies[bra_state]
            if abs(dE) < 1.0e-14:
                dE = m.copysign(1.0e-14, dE)
            out /= dE
        return out

    ## returns \f$ \sum_\alpha v^\alpha D^\alpha \f$ where \f$ D^\alpha_{ij} = d^\alpha_{ij} \f$
    def compute_NAC_matrix(self, velocity):
        nstates = self.nstates()
        ndim = self.ndim()
        assert(ndim == velocity.shape[0])

        # first build a contraction of dV with the velocity
        dV = np.dot(self.dV.reshape([ndim, nstates*nstates]).T, velocity).reshape([nstates,nstates])

        # transform to eigenbasis
        out = np.dot(self.coeff.T, np.dot(dV, self.coeff))

        for i in range(nstates):
            # build dE inverse all at once
            dE = self.energies[0:i] - self.energies[i]
            # replace exact zeros
            dE[dE==0.0] = 1.0e-30

            # still use clip to keep results in reasonable window
            dE = np.reciprocal(dE).clip(-1.0e14, 1.0e14)

            out[i,0:i] *= dE
            out[0:i,i] = -out[i,0:i]

            out[i,i] = 0.0
        return out

    ## returns a single column of \f$ \sum_\alpha v^\alpha D^\alpha \f$ where \f$ D^\alpha_{ij} = d^\alpha_{ij} \f$
    ## and \f$i\f$ is specified
    def compute_NAC_column(self, velocity, i):
        nstates = self.nstates()
        ndim = self.ndim()
        assert(ndim == velocity.shape[0])

        # first build a contraction of dV with the velocity
        dV = np.dot(self.dV.reshape([ndim, nstates*nstates]).T, velocity).reshape([nstates,nstates])

        # transform to eigenbasis in [i,:] type form
        out = np.dot(self.coeff[:,i].T, np.dot(dV, self.coeff))

        dE = self.energies[:] - self.energies[i]
        # replace exact zeros with small number
        dE[dE==0.0] = 1.0e-30

        # divide by reciprocal clip
        out *= np.reciprocal(dE).clip(-1.0e14,1.0e14)

        # but now fix diagonal
        out[i] = 0.0

        return out

## Class to propagate a single SH Trajectory
class TrajectorySH:
    def __init__(self, model, tracer, **options):
        self.model = model
        self.tracer = tracer
        self.mass = options["mass"]
        self.position = options["position"]
        self.velocity = options["momentum"] / self.mass
        self.last_velocity = np.zeros([model.ndim()])
        if options["initial_state"] == "ground":
            self.rho = np.zeros([model.nstates(),model.nstates()], dtype=np.complex128)
            self.rho[0,0] = 1.0
            self.state = 0
        else:
            raise Exception("Unrecognized initial state option")

        # check_end must be a class that implements __call__ that accepts TrajectorySH
        # and returns True when the simulation is over
        if "check_end" in options:
            self.check_end = options["check_end"]()
        else:
            self.check_end = end_checker(abs(options["position"]), 1e30)()

        # fixed initial parameters
        self.time = 0.0

        # read out of options
        self.dt = options["dt"]
        self.outcome_type = options["outcome_type"]

        # propagator
        self.propagator = options["propagator"]

        self.random = np.random.RandomState(options["seed"])

    def trace(self, electronics, prob, call):
        if call == "collect":
            func = self.tracer.collect
        elif call == "finalize":
            func = self.tracer.finalize

        func(self.time, np.copy(self.position), self.mass*np.copy(self.velocity), np.copy(self.rho), self.state, electronics, prob)

    def kinetic_energy(self):
        return 0.5 * self.mass * np.dot(self.velocity, self.velocity)

    def total_energy(self, elec_state):
        potential = elec_state.compute_potential(self.state)
        kinetic = self.kinetic_energy()
        return potential + kinetic

    def mode_kinetic_energy(self, direction):
        component = np.dot(direction, self.velocity) / np.dot(direction, direction) * direction
        return 0.5 * self.mass * np.dot(component, component)

    ## Rescales velocity in the specified direction and amount
    # @param direction array specifying the direction of the velocity to rescale
    # @param reduction scalar specifying how much kinetic energy should be damped
    def rescale_component(self, direction, reduction):
        # normalize
        direction /= np.sqrt(np.dot(direction, direction))
        Md = self.mass * direction
        a = np.dot(Md, Md)
        b = 2.0 * np.dot(self.mass * self.velocity, Md)
        c = -2.0 * self.mass * reduction
        roots = np.roots([a, b, c])
        scal = min(roots, key=lambda x: abs(x))
        self.velocity += scal * direction

    ## Propagates \f$\rho(t)\f$ to \f$\rho(t + dt)\f$
    # @param elec_states ElectronicStates at \f$t\f$
    #
    # The propagation assumes the electronic energies and couplings are static throughout.
    # This will only be true for fairly small time steps
    def propagate_rho(self, elec_states, dt):
        velo = 0.5 * (self.last_velocity + self.velocity)

        W = np.diag(elec_states.energies) - 1j * elec_states.compute_NAC_matrix(velo)

        if self.propagator == "exponential":
            diags, coeff = np.linalg.eigh(W)

            # use W as temporary storage
            U = np.dot(coeff, np.dot(np.diag(np.exp(-1j * diags * dt)), coeff.T.conj(), out=W))
            np.dot(U, np.dot(self.rho, U.T.conj(), out=W), out=self.rho)
        elif self.propagator == "ode":
            nstates = self.model.nstates()
            G *= -1j
            def drho(time, y):
                ymat = np.reshape(y, [nstates, nstates])
                dro = np.dot(G, ymat) - np.dot(ymat, G)
                return np.reshape(dro, [nstates*nstates])

            rhovec = np.reshape(self.rho, [nstates*nstates])
            integrator = scipy.integrate.complex_ode(drho).set_integrator('vode', method='bdf', with_jacobian=False)
            integrator.set_initial_value(rhovec, self.time)
            integrator.integrate(self.time + dt)
            self.rho = np.reshape(integrator.y, [nstates,nstates])
            if not integrator.successful():
                exit("Propagation of the electronic wavefunction failed!")
        else:
            raise Exception("Unrecognized method for propagation of electronic density matrix!")

    ## Compute probability of hopping, generate random number, and perform hops
    def surface_hopping(self, elec_states):
        nstates = self.model.nstates()

        velo = 0.5 * (self.last_velocity + self.velocity) # interpolate velocities to get value at integer step
        W = elec_states.compute_NAC_column(velo, self.state)

        probs = 2.0 * np.real(self.rho[self.state,:]) * W[:] * self.dt / np.real(self.rho[self.state,self.state])

        # zero out 'self-hop' for good measure (numerical safety)
        probs[self.state] = 0.0

        # clip probabilities to make sure they are between zero and one
        probs = probs.clip(0.0, 1.0)

        accumulated_P = np.cumsum(probs)
        zeta = self.random.uniform()
        do_hop = np.less(zeta, accumulated_P)
        if (any(do_hop)): # do switch
            # jump to the first state for which zeta is less
            for target in range(nstates):
                if do_hop[target]: break

            new_potential, old_potential = elec_states.energies[target], elec_states.energies[self.state]
            delV = new_potential - old_potential
            derivative_coupling = elec_states.compute_derivative_coupling(target, self.state)
            component_kinetic = self.mode_kinetic_energy(derivative_coupling)
            if delV <= component_kinetic:
                self.state = target
                self.rescale_component(derivative_coupling, -delV)
                self.tracer.hops += 1
        return sum(probs)

    ## helper function to simplify the calculation of the electronic states at a given position
    def compute_electronics(self, position, ref_coeff = None):
        return ElectronicStates(model.V(position), model.dV(position), ref_coeff)

    ## run simulation
    def simulate(self):
        last_electronics = None
        electronics = self.compute_electronics(self.position)

        # start by taking half step in velocity
        initial_acc = electronics.compute_force(self.state) / self.mass
        veloc = self.velocity
        dv = 0.5 * initial_acc * self.dt
        self.last_velocity, self.velocity = veloc - dv, veloc + dv

        # propagate wavefunction a half-step forward to match velocity
        self.propagate_rho(electronics, 0.5*self.dt)
        potential_energy = electronics.compute_potential(self.state)

        prob = 0.0

        self.trace(electronics, prob, "collect")

        # propagation
        while (True):
            # first update nuclear coordinates
            self.position += self.velocity * self.dt

            # calculate electronics at new position
            last_electronics, electronics = electronics, self.compute_electronics(self.position, electronics.coeff)
            acceleration = electronics.compute_force(self.state) / self.mass
            self.last_velocity, self.velocity = self.velocity, self.velocity + acceleration * self.dt

            # now propagate the electronic wavefunction to the new time
            self.propagate_rho(electronics, self.dt)
            prob = self.surface_hopping(electronics)
            self.time += self.dt

            # ending condition
            if (self.check_end(self)):
                break

            self.trace(electronics, prob, "collect")

        self.trace(electronics, prob, "finalize")

        return self.tracer

    ## Classifies end of simulation:
    #
    #  2*state + [0 for left, 1 for right]
    def outcome(self):
        out = np.zeros([self.model.nstates(), 2])
        lr = 0 if self.position < 0.0 else 1
        if self.outcome_type == "populations":
            out[:,lr] = np.real(self.rho).diag()[:]
        elif self.outcome_type == "state":
            out[self.state,lr] = 1.0
        else:
            raise Exception("Unrecognized outcome recognition type")
        return out
        # first bit is left (0) or right (1), second bit is electronic state

## Class to collect observables for a given trajectory
TraceData = collections.namedtuple('TraceData', 'time position momentum rho activestate electronics hopping')

class Trace:
    def __init__(self):
        self.data = []
        self.hops = 0

    ## collect and optionally process data
    def collect(self, time, position, momentum, rho, activestate, electronics, prob):
        self.data.append(TraceData(time=time, position=position, momentum=momentum,
                                rho=rho, activestate=activestate, electronics=electronics, hopping=prob))

    ## finalize an individual trajectory
    def finalize(self, time, position, momentum, rho, activestate, electronics, prob):
        self.collect(time, position, momentum, rho, activestate, electronics, prob)

    def __iter__(self):
        return self.data.__iter__()

## Class to manage the collection of observables from a set of trajectories
class TraceManager:
    def __init__(self):
        self.traces = []

    ## returns a Tracer object that will collect all of the observables for a given
    #  trajectory
    def spawn_tracer(self):
        return Trace()

    ## accepts a Tracer object and adds it to list of traces
    def merge_tracer(self, tracer):
        self.traces.append(tracer.data)

    ## merge other manager into self
    def add_batch(self, traces):
        self.traces.extend(traces)

    def __iter__(self):
        return self.traces.__iter__()

## Canned function that returns a class to check for the end of a simulation
def end_checker(box_bounds, nsteps):
    class CheckEnd(object):
        def __init__(self):
            self.reached_interaction = False

        def __call__(self, traj):
            if self.reached_interaction: # simulation has made it to interaction region
                if traj.time > traj.dt * nsteps:
                    return True
                else:
                    return traj.position < -box_bounds or traj.position > box_bounds
            else: # check whether in interaction region
                if traj.position > -box_bounds and traj.position < box_bounds:
                    self.reached_interaction = True
                return False

    return CheckEnd

#####################################################################################
# Series of canned functions that return generator functions for initial conditions #
#####################################################################################

## Canned function that returns a generator function that generates initial conditions
## for batches of trajectories.
def const_init(initial_position, initial_momentum, initial_rho):
    def gen_wrapper(nsamples):
        for i in range(nsamples):
            yield { "position" : initial_position, "momentum" : initial_momentum, "initial_state" : initial_rho }
    return gen_wrapper

## Class to manage many FSSH trajectories
#
# Requires a model object which is a class that has functions V(x), dV(x), nstates(), and ndim()
# that return the Hamiltonian at position x, gradient of the Hamiltonian at position x
# number of electronic states, and dimension of nuclear space, respectively.
class BatchedTraj:
    ## Constructor requires model and options input as kwargs
    # @param model object used to describe the model system
    #
    # Accepted keyword arguments and their defaults:
    # | key                |   default                  |
    # ---------------------|----------------------------|
    # | mass               | 2000.0                     |
    # | initial_time       | 0.0                        |
    # | samples            | 2000                       |
    # | dt                 | 20.0  ~ 0.5 fs            |
    # | propagator         | "exponential"              |
    # | nprocs             | MultiProcessing.cpu_count  |
    # | outcome_type       | "state"                    |
    def __init__(self, model, tracemanager = TraceManager(), **inp):
        self.model = model
        self.tracemanager = tracemanager
        self.options = {}

        # system parameters
        self.options["mass"]          = inp.get("mass", 2000.0)

        # time parameters
        self.options["initial_time"]  = inp.get("initial_time", 0.0)
        # statistical parameters
        self.options["samples"]       = inp.get("samples", 2000)
        self.options["dt"]            = inp.get("dt", 20.0) # default to roughly half a femtosecond

        # random seed
        self.options["seed"]          = inp.get("seed", None)

        # numerical parameters
        self.options["propagator"]    = inp.get("propagator", "exponential")
        if self.options["propagator"] not in ["exponential", "ode"]:
            raise Exception("Unrecognized electronic propagator!")
        elif self.options["propagator"] == "ode":
            try:
                import scipy.integrate
            except ImportError:
                print "Error: scipy is required for the ode propagator!"
                quit()

        self.options["nprocs"]        = inp.get("nprocs", mp.cpu_count())
        self.options["outcome_type"]  = inp.get("outcome_type", "state")

        if "check_end" in inp:
            self.options["check_end"] = inp["check_end"]
        else:
            raise Exception("A class must be provided that will check for the end of a simulation: check_end")

        if "traj_gen" in inp:
            self.options["traj_gen"] = inp["traj_gen"]
        else:
            raise Exception("A generator function must be provided that generates initial conditions: traj_gen")

    ## runs a set of trajectories and collects the results
    # @param n number of trajectories to run
    def run_trajectories(self, n):
        outcomes = np.zeros([self.model.nstates(),2])
        traces = []
        try:
            for params in self.options["traj_gen"](n):
                traj_input = self.options
                traj_input.update(params)
                traj = TrajectorySH(self.model, self.tracemanager.spawn_tracer(), **traj_input)
                trace = traj.simulate()
                traces.append(trace)
                outcomes += traj.outcome()
            return (outcomes, traces)
        except KeyboardInterrupt:
            pass

    ## runs many trajectories and returns averaged results
    def compute(self):
        # for now, define four possible outcomes of the simulation
        outcomes = np.zeros([self.model.nstates(),2])
        nsamples = int(self.options["samples"])
        energy_list = []
        nprocs = self.options["nprocs"]

        if nprocs > 1:
            pool = mp.Pool(nprocs)
            chunksize = (nsamples - 1)/nprocs + 1
            batches = [ min(chunksize, nsamples - chunksize*ip) for ip in range(nprocs) ]
            poolresult = [ pool.apply_async(unwrapped_run_trajectories, (self, b)) for b in batches ]
            try:
                for r in poolresult:
                    oc, tr = r.get()
                    outcomes += oc
                    self.tracemanager.add_batch(tr)
            except KeyboardInterrupt:
                pool.terminate()
                pool.join()
                exit(" Aborting!")
            pool.close()
            pool.join()
        else:
            try:
                oc, tr = unwrapped_run_trajectories(self, nsamples)
                outcomes += oc
                self.tracemanager.add_batch(tr)
            except KeyboardInterrupt:
                exit(" Aborting!")

        outcomes /= float(nsamples)
        self.tracemanager.outcomes = outcomes
        return self.tracemanager

## global version of BatchedTraj.run_trajectories that is necessary because of the stupid way threading pools work in python
def unwrapped_run_trajectories(fssh, n):
    return BatchedTraj.run_trajectories(fssh, n)

if __name__ == "__main__":
    import tullymodels as tm
    import argparse as ap

    parser = ap.ArgumentParser(description="Example driver for FSSH")

    parser.add_argument('-m', '--model', default='simple', choices=[x for x in tm.modeldict], type=str, help="Tully model to plot (%(default)s)")
    parser.add_argument('-k', '--krange', default=(0.1,30.0), nargs=2, type=float, help="range of momenta to consider (%(default)s)")
    parser.add_argument('-n', '--nk', default=20, type=int, help="number of momenta to compute (%(default)d)")
    parser.add_argument('-l', '--kspacing', default="linear", type=str, choices=('linear', 'log'), help="linear or log spacing for momenta (%(default)s)")
    parser.add_argument('-s', '--samples', default=200, type=int, help="number of samples (%(default)d)")
    parser.add_argument('-j', '--nprocs', default=2, type=int, help="number of processors (%(default)d)")
    parser.add_argument('-p', '--propagator', default="exponential", choices=('exponential', 'ode'), type=str, help="propagator (%(default)s)")
    parser.add_argument('-M', '--mass', default=2000.0, type=float, help="particle mass (%(default)s)")
    parser.add_argument('-t', '--dt', default=None, type=float, help="time step (%(default)s)")
    parser.add_argument('-T', '--nt', default=2000, type=int, help="max number of steps (%(default)s)")
    parser.add_argument('-x', '--position', default=-10.0, type=float, help="starting position (%(default)s)")
    parser.add_argument('-b', '--bounds', default=5.0, type=float, help="bounding box to end simulation (%(default)s)")
    parser.add_argument('-o', '--output', default="averaged", type=str, help="what to print as output (%(default)s)")
    parser.add_argument('-z', '--seed', default=None, type=int, help="random seed (current date)")
    parser.add_argument('--published', dest="published", action="store_true", help="override ranges to use those found in relevant papers (%(default)s)")

    args = parser.parse_args()

    if args.model in tm.modeldict:
        model = tm.modeldict[args.model]()
    else:
        raise Exception("Unknown model chosen") # the argument parser should prevent this throw from being possible

    if (args.seed is not None):
        np.random.seed(args.seed)

    nk = args.nk
    min_k, max_k = args.krange

    if (args.published): # hack spacing to resemble Tully's
        if (args.model == "simple"):
            min_k, max_k = 1.0, 35.0
        elif (args.model == "dual"):
            min_k, max_k = m.log10(m.sqrt(2.0 * args.mass * m.exp(-4.0))), m.log10(m.sqrt(2.0 * args.mass * m.exp(1.0)))
        elif (args.model == "extended"):
            min_k, max_k = 1.0, 35.0
        elif (args.model == "super"):
            min_k, max_k = 0.5, 20.0
        else:
            print "Warning! published option chosen but no available bounds! Using inputs."

    kpoints = []
    if args.kspacing == "linear":
        kpoints = np.linspace(min_k, max_k, nk)
    elif args.kspacing == "log":
        kpoints = np.logspace(min_k, max_k, nk)
    else:
        raise Exception("Unrecognized type of spacing")

    for k in kpoints:
        fssh = BatchedTraj(model, momentum = k,
                           position = args.position,
                           mass = args.mass,
                           samples = args.samples,
                           propagator = args.propagator,
                           nprocs = args.nprocs,
                           dt = args.dt,
                           seed = args.seed,
                           traj_gen = const_init(args.position, k, "ground"),
                           check_end = end_checker(args.bounds, args.nt)
                   )
        results = fssh.compute()
        outcomes = results.outcomes

        if (args.output == "averaged"):
            print "%12.6f %s" % (k, " ".join(["%12.6f" % x for x in np.nditer(outcomes)]))
        elif (args.output == "single"):
            for i in results.traces[0]:
                print "%12.6f %12.6f %12.6f %6d" % (i.time, i.position, i.momentum, i.activestate)
        else:
            print "Not printing results. This is probably not what you wanted!"
