#!/usr/bin/env python

## @package fssh
#  Module responsible for propagating surface hopping trajectories
from __future__ import print_function, division

import numpy as np
import math as m
import multiprocessing as mp
import collections
import pickle

import sys

## Wrapper around all the information computed for a set of electronics
#  states at a given position: V, dV, eigenvectors, eigenvalues
#  Parameters with names starting with '_' indicate things that by
#  design should not be called outside of ElectronicStates.
class ElectronicStates(object):
    ## Constructor
    # @param V Hamiltonian/potential
    # @param dV Gradient of Hamiltonian/potential
    # @param ref_coeff [optional] set of coefficients (for example, from a previous step) used to keep the sign of eigenvectors consistent
    def __init__(self, V, dV, reference = None):
        # raw internal quantities
        self._V = V
        self._dV = dV
        self._coeff = self._compute_coeffs(reference)

        # processed quantities used in simulation
        self.hamiltonian = self._compute_hamiltonian()
        self.force = self._compute_force()
        self.derivative_coupling = self._compute_derivative_coupling()

    ## returns dimension of (electronic) Hamiltonian
    def nstates(self):
        return self._V.shape[1]

    ## returns dimensionality of model (nuclear coordinates)
    def ndim(self):
        return self._dV.shape[0]

    ## returns coefficient matrix for basis states
    def _compute_coeffs(self, reference):
        energies, coeff = np.linalg.eigh(self._V)
        if reference is not None:
            try:
                ref_coeff = reference._coeff
                for mo in range(self.nstates()):
                    if (np.dot(coeff[:,mo], ref_coeff[:,mo]) < 0.0):
                        coeff[:,mo] *= -1.0
            except:
                raise Exception("Failed to regularize new ElectronicStates from a reference object %s" % (reference))
        self._energies = energies
        return coeff

    ## returns \f$-\langle \phi_{\mbox{state}} | \nabla H | \phi_{\mbox{state}} \rangle\f$ of Hamiltonian
    def _compute_force(self):
        out = np.zeros([self.nstates(), self.ndim()])
        for ist in range(self.nstates()):
            state_vec = self._coeff[:,ist]
            for d in range(self.ndim()):
                out[ist,d] = - (np.dot(state_vec.T, np.dot(self._dV[d,:,:], state_vec)))
        return out

    ## returns \f$\phi_{\mbox{state}} | H | \phi_{\mbox{state}} = \varepsilon_{\mbox{state}}\f$
    def _compute_hamiltonian(self):
        return np.dot(self._coeff.T, np.dot(self._V, self._coeff))

    ## returns \f$\phi_{i} | \nabla_\alpha \phi_{j} = d^\alpha_{ij}\f$
    def _compute_derivative_coupling(self):
        out = np.zeros([self.nstates(), self.nstates(), self.ndim()])
        for j in range(self.nstates()):
            for i in range(j):
                for d in range(self.ndim()):
                    out[i,j,d] = np.dot(self._coeff[:,i].T, np.dot(self._dV[d,:,:], self._coeff[:,j]))
                dE = self._energies[j] - self._energies[i]
                if abs(dE) < 1.0e-14:
                    dE = m.copysign(1.0e-14, dE)

                out[i,j,:] /= dE
                out[j,i,:] = -out[i,j,:]

        return out

    ## returns \f$ \sum_\alpha v^\alpha D^\alpha \f$ where \f$ D^\alpha_{ij} = d^\alpha_{ij} \f$
    def NAC_matrix(self, velocity):
        nstates = self.nstates()
        ndim = self.ndim()
        assert(ndim == velocity.shape[0])

        return np.dot(self.derivative_coupling, velocity)

## Class to propagate a single SH Trajectory
class TrajectorySH(object):
    def __init__(self, model, tracer, **options):
        self.model = model
        self.tracer = tracer
        self.mass = options["mass"]
        self.position = np.array(options["position"]).reshape(model.ndim())
        self.velocity = np.array(options["momentum"]).reshape(model.ndim()) / self.mass
        self.last_velocity = np.zeros([model.ndim()])
        if options["initial_state"] == "ground":
            self.rho = np.zeros([model.nstates(),model.nstates()], dtype=np.complex128)
            self.rho[0,0] = 1.0
            self.state = 0
        else:
            raise Exception("Unrecognized initial state option")

        # function duration_initialize should get us ready to for future continue_simulating calls
        # that decide whether the simulation has finished
        self.duration_initialize()

        # fixed initial parameters
        self.time = 0.0
        self.nsteps = 0

        # read out of options
        self.dt = options["dt"]
        self.outcome_type = options["outcome_type"]

        self.random_state = np.random.RandomState(options["seed"])

    def random(self):
        return self.random_state.uniform()

    def currently_interacting(self):
        """determines whether trajectory is currently inside an interaction region"""
        return self.box_bounds[0] < self.position and self.box_bounds[1] > self.position

    def duration_initialize(self):
        """Initializes variables related to continue_simulating"""
        self.found_box = False
        self.box_bounds = (-5,5)
        self.max_steps = 10000

    def continue_simulating(self):
        """Returns True if a trajectory ought to keep running, False if it should finish"""
        if self.nsteps > self.max_steps:
            return False
        elif self.found_box:
            return self.currently_interacting()
        else:
            if self.currently_interacting():
                self.found_box = True
            return True

    def trace(self, electronics, prob, call):
        if call == "collect":
            func = self.tracer.collect
        elif call == "finalize":
            func = self.tracer.finalize

        func(self.time, np.copy(self.position), self.mass*np.copy(self.velocity), np.copy(self.rho), self.state, electronics, prob)

    def kinetic_energy(self):
        return 0.5 * self.mass * np.dot(self.velocity, self.velocity)

    def potential_energy(self, electronics):
        return electronics.hamiltonian[self.state,self.state]

    def total_energy(self, electronics):
        potential = self.potential_energy(electronics)
        kinetic = self.kinetic_energy()
        return potential + kinetic

    def force(self, electronics):
        return electronics.force[self.state,:]

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

        W = elec_states.hamiltonian - 1j * elec_states.NAC_matrix(velo)

        diags, coeff = np.linalg.eigh(W)

        # use W as temporary storage
        U = np.linalg.multi_dot([ coeff, np.diag(np.exp(-1j * diags * dt)), coeff.T.conj() ])
        np.dot(U, np.dot(self.rho, U.T.conj(), out=W), out=self.rho)

    ## Compute probability of hopping, generate random number, and perform hops
    def surface_hopping(self, elec_states):
        nstates = self.model.nstates()

        velo = 0.5 * (self.last_velocity + self.velocity) # interpolate velocities to get value at integer step
        W = elec_states.NAC_matrix(velo)[self.state, :]

        probs = 2.0 * np.real(self.rho[self.state,:]) * W[:] * self.dt / np.real(self.rho[self.state,self.state])

        # zero out 'self-hop' for good measure (numerical safety)
        probs[self.state] = 0.0

        # clip probabilities to make sure they are between zero and one
        probs = probs.clip(0.0, 1.0)

        do_hop, hop_to = self.hopper(probs)
        if (do_hop): # do switch
            new_potential, old_potential = elec_states.hamiltonian[hop_to, hop_to], elec_states.hamiltonian[self.state, self.state]
            delV = new_potential - old_potential
            derivative_coupling = elec_states.derivative_coupling[hop_to, self.state, :]
            component_kinetic = self.mode_kinetic_energy(derivative_coupling)
            if delV <= component_kinetic:
                self.state = hop_to
                self.rescale_component(derivative_coupling, -delV)
                self.tracer.hops += 1

        return sum(probs)

    ## given a set of probabilities, determines whether and where to hop
    def hopper(self, probs):
        zeta = self.random()
        acc_prob = np.cumsum(probs)
        hops = np.less(zeta, acc_prob)
        if any(hops):
            hop_to = -1
            for i in range(self.model.nstates()):
                if hops[i]:
                    hop_to = i
                    break

            return True, hop_to
        else:
            return False, -1

    ## helper function to simplify the calculation of the electronic states at a given position
    def compute_electronics(self, position, electronics = None):
        return ElectronicStates(self.model.V(position), self.model.dV(position), electronics)

    ## run simulation
    def simulate(self):
        last_electronics = None
        electronics = self.compute_electronics(self.position)

        # start by taking half step in velocity
        initial_acc = self.force(electronics) / self.mass
        veloc = self.velocity
        dv = 0.5 * initial_acc * self.dt
        self.last_velocity, self.velocity = veloc - dv, veloc + dv

        # propagate wavefunction a half-step forward to match velocity
        self.propagate_rho(electronics, 0.5*self.dt)
        potential_energy = self.potential_energy(electronics)

        prob = 0.0

        self.trace(electronics, prob, "collect")

        # propagation
        while (True):
            # first update nuclear coordinates
            self.position += self.velocity * self.dt

            # calculate electronics at new position
            last_electronics, electronics = electronics, self.compute_electronics(self.position, electronics)
            acceleration = self.force(electronics) / self.mass
            self.last_velocity, self.velocity = self.velocity, self.velocity + acceleration * self.dt

            # now propagate the electronic wavefunction to the new time
            self.propagate_rho(electronics, self.dt)
            prob = self.surface_hopping(electronics)
            self.time += self.dt
            self.nsteps += 1

            # ending condition
            if not self.continue_simulating():
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

class Trace(object):
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

    def __getitem__(self, i):
        return self.data[i]

## Class to manage the collection of observables from a set of trajectories
class TraceManager(object):
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

    def __getitem__(self, i):
        return self.traces[i]

## Canned class that checks for the end of a simulation.
## Requires one to directly manipulate the class parameters to change the bounds and steps allowed

class StillInteracting(Exception):
    def __init__(self):
        Exception.__init__(self, "A simulation ended while still inside the interaction region.")

#####################################################################################
# Series of canned classes act as generator functions for initial conditions        #
#####################################################################################

## Canned class whose call function acts as a generator for static initial conditions
class TrajGenConst(object):
    def __init__(self, position, momentum, initial_state):
        self.position = position
        self.momentum = momentum
        self.initial_state = initial_state

    def __call__(self, nsamples):
        for i in range(nsamples):
            yield { "position" : self.position, "momentum" : self.momentum, "initial_state" : self.initial_state }

## Canned class whose call function acts as a generator for normally distributed initial conditions
class TrajGenNormal(object):
    def __init__(self, position, momentum, initial_state, sigma, seed = None):
        self.position = position
        self.position_deviation = 0.5 * sigma
        self.momentum = momentum
        self.momentum_deviation = 1.0 / sigma
        self.initial_state = initial_state

        self.random_state = np.random.RandomState(seed)

    def kskip(self, ktest):
        return ktest < 0.0

    def __call__(self, nsamples):
        for i in range(nsamples):
            x = self.random_state.normal(self.position, self.position_deviation)
            k = self.random_state.normal(self.momentum, self.momentum_deviation)

            if (self.kskip(k)): continue
            yield { "position": x, "momentum": k, "initial_state": self.initial_state }

## Class to manage many TrajectorySH trajectories
#
# Requires a model object which is a class that has functions V(x), dV(x), nstates(), and ndim()
# that return the Hamiltonian at position x, gradient of the Hamiltonian at position x
# number of electronic states, and dimension of nuclear space, respectively.
class BatchedTraj(object):
    ## Constructor requires model and options input as kwargs
    # @param model object used to describe the model system
    #
    # Accepted keyword arguments and their defaults:
    # | key                |   default                  |
    # ---------------------|----------------------------|
    # | mass               | 2000.0                     |
    # | initial_time       | 0.0                        |
    # | samples            | 2000                       |
    # | dt                 | 20.0  ~ 0.5 fs             |
    # | seed               | None (date)                |
    # | nprocs             | MultiProcessing.cpu_count  |
    # | outcome_type       | "state"                    |
    def __init__(self, model, traj_gen, trajectory_type = TrajectorySH, tracemanager = TraceManager(), **inp):
        self.model = model
        self.tracemanager = tracemanager
        self.trajectory = trajectory_type
        self.traj_gen = traj_gen
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

        self.options["nprocs"]        = inp.get("nprocs", 1)
        self.options["outcome_type"]  = inp.get("outcome_type", "state")

    ## runs a set of trajectories and collects the results
    # @param n number of trajectories to run
    def run_trajectories(self, n):
        outcomes = np.zeros([self.model.nstates(),2])
        traces = []
        try:
            for params in self.traj_gen(n):
                traj_input = self.options
                traj_input.update(params)
                traj = self.trajectory(self.model, self.tracemanager.spawn_tracer(), **traj_input)
                try:
                    trace = traj.simulate()
                    traces.append(trace)
                    outcomes += traj.outcome()
                except StillInteracting:
                    print("BEWARE: a simulation ended while still in interaction region", file=sys.stderr)
                    pass
            return (outcomes, traces)
        except KeyboardInterrupt:
            raise

    ## runs many trajectories and returns averaged results
    def compute(self):
        # for now, define four possible outcomes of the simulation
        outcomes = np.zeros([self.model.nstates(),2])
        nsamples = int(self.options["samples"])
        energy_list = []
        nprocs = self.options["nprocs"]

        if nprocs > 1:
            pool = mp.Pool(nprocs)
            chunksize = min((nsamples - 1)//nprocs + 1, 5)
            nchunks = (nsamples -1)//chunksize + 1
            batches = [ min(chunksize, nsamples - chunksize*ip) for ip in range(nchunks) ]
            poolresult = [ pool.apply_async(unwrapped_run_trajectories, (self, b)) for b in batches ]
            try:
                for r in poolresult:
                    oc, tr = r.get()
                    outcomes += oc
                    self.tracemanager.add_batch(tr)
            except KeyboardInterrupt:
                return
                pool.terminate()
                pool.join()
                exit(" Aborting!")
            pool.close()
            pool.join()
        else:
            try:
                oc, tr = self.run_trajectories(nsamples)
                outcomes += oc
                self.tracemanager.add_batch(tr)
            except KeyboardInterrupt:
                exit(" Aborting!")

        outcomes /= np.sum(outcomes)
        self.tracemanager.outcomes = outcomes
        return self.tracemanager

## global version of BatchedTraj.run_trajectories that is necessary because of the stupid way threading pools work in python
def unwrapped_run_trajectories(fssh, n):
    try:
        return BatchedTraj.run_trajectories(fssh, n)
    except KeyboardInterrupt:
        pass

class TrajectoryCum(TrajectorySH):
    def __init__(self, *args, **kwargs):
        TrajectorySH.__init__(self, *args, **kwargs)

        self.prob_cum = 0.0
        self.zeta = self.random()

    def hopper(self, probs):
        accumulated = self.prob_cum
        for i, p in enumerate(probs):
            accumulated = accumulated + (1 - accumulated) * p
            if accumulated > self.zeta: # then hop
                # reset prob_cum, zeta
                self.prob_cum = 0.0
                self.zeta = self.random()
                return True, i

        self.prob_cum = accumulated

        return False, -1

class Ehrenfest(TrajectorySH):
    def __init__(self, *args, **kwargs):
        TrajectorySH.__init__(self, *args, **kwargs)

    def potential_energy(self, electronics):
        return np.real(np.dot(self.rho, electronics.hamiltonian))

    def force(self, electronics):
        return np.dot(np.real(np.diag(self.rho)), electronics.force)

    def surface_hopping(self, electronics):
        return 0.0

if __name__ == "__main__":
    import tullymodels as tm
    import argparse as ap

    parser = ap.ArgumentParser(description="Example driver for SH")

    parser.add_argument('-a', '--method', default="fssh", choices=("fssh", "fssh-cumulative", "ehrenfest"), type=str.lower, help="Variant of SH")
    parser.add_argument('-m', '--model', default='simple', choices=[x for x in tm.modeldict], type=str, help="Tully model to plot (%(default)s)")
    parser.add_argument('-k', '--krange', default=(0.1,30.0), nargs=2, type=float, help="range of momenta to consider (%(default)s)")
    parser.add_argument('-n', '--nk', default=20, type=int, help="number of momenta to compute (%(default)d)")
    parser.add_argument('-l', '--kspacing', default="linear", type=str, choices=('linear', 'log'), help="linear or log spacing for momenta (%(default)s)")
    parser.add_argument('-K', '--ksampling', default="none", type=str, choices=('none', 'normal'), help="how to sample momenta for a set of simulations (%(default)s)")
    parser.add_argument('-f', '--normal', default=20, type=float, help="standard deviation as a proportion of inverse momentum for normal samping (%(default)s)")
    parser.add_argument('-s', '--samples', default=200, type=int, help="number of samples (%(default)d)")
    parser.add_argument('-j', '--nprocs', default=2, type=int, help="number of processors (%(default)d)")
    parser.add_argument('-M', '--mass', default=2000.0, type=float, help="particle mass (%(default)s)")
    parser.add_argument('-t', '--dt', default=20.0, type=float, help="time step in a.u.(%(default)s)")
    parser.add_argument('-y', '--scale_dt', dest="scale_dt", action="store_true", help="scale (hack-like) time step using momentum (%(default)s)")
    parser.add_argument('-T', '--nt', default=50000, type=int, help="max number of steps (%(default)s)")
    parser.add_argument('-x', '--position', default=-10.0, type=float, help="starting position (%(default)s)")
    parser.add_argument('-b', '--bounds', default=5.0, type=float, help="bounding box to end simulation (%(default)s)")
    parser.add_argument('-o', '--output', default="averaged", type=str, choices=('averaged', 'single', 'pickle', 'swarm', 'hack'), help="what to produce as output (%(default)s)")
    parser.add_argument('-O', '--outfile', default="sh.pickle", type=str, help="name of pickled file to produce (%(default)s)")
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
            print("Warning! Published option chosen but no available bounds! Using inputs.", file=sys.stderr)

    kpoints = []
    if args.kspacing == "linear":
        kpoints = np.linspace(min_k, max_k, nk)
    elif args.kspacing == "log":
        kpoints = np.logspace(min_k, max_k, nk)
    else:
        raise Exception("Unrecognized type of spacing")

    if args.method == "fssh":
        trajectory_type = TrajectorySH
    elif args.method == "fssh-cumulative":
        trajectory_type = TrajectoryCum
    elif args.method == "ehrenfest":
        trajectory_type = Ehrenfest

    all_results = []

    for k in kpoints:
        if args.ksampling == "none":
            traj_gen = TrajGenConst(args.position, k, "ground")
        elif args.ksampling == "normal":
            traj_gen = TrajGenNormal(args.position, k, "ground", sigma = args.normal/k)

        # hack-y scale of time step so that the input amount roughly makes sense for 10.0 a.u.
        dt = args.dt * (10.0 / k) if args.scale_dt else args.dt

        fssh = BatchedTraj(model, traj_gen,
                           trajectory_type = trajectory_type,
                           momentum = k,
                           position = args.position,
                           mass = args.mass,
                           samples = args.samples,
                           nprocs = args.nprocs,
                           dt = dt,
                           seed = args.seed
                   )
        results = fssh.compute()
        outcomes = results.outcomes

        if (args.output == "single"):
            for i in results.traces[0]:
                line = "%12.6f %12.6f %12.6f " % (i.time, i.position, i.momentum)
                line += " ".join(["%12.6f" % x for x in np.real(np.diag(i.rho))])
                line += " %6d" % i.activestate
                print(line)
        elif (args.output == "swarm"):
            maxsteps = max([ len(t) for t in results.traces ])
            for i in range(maxsteps):
                for t in results.traces:
                    if i < len(t):
                        print("%12.6f %12.6f %6d" % (t[i].time, t[i].position, t[i].activestate))
                print()
                print()
        elif (args.output == "averaged" or args.output == "pickle"):
            print("%12.6f %s" % (k, " ".join(["%12.6f" % x for x in np.nditer(outcomes)])))
            if (args.output == "pickle"): # save results for later processing
                all_results.append((k, results))
        elif (args.output == "hack"):
            print("Put your very own output functions here.")
        else:
            print("Not printing results. This is probably not what you wanted!")

    if (len(all_results) > 0):
        pickle.dump(all_results, open(args.outfile, "wb"))
