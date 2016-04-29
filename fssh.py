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
            dE = self.energies[ket_state] - self.energies[bra_state]
            if abs(dE) < 1.0e-14:
                dE = m.copysign(1.0e-14, dE)

            for d in range(self.ndim()):
                out[d] = np.dot(self.coeff[:,bra_state].T, np.dot(self.dV[d,:,:], self.coeff[:,ket_state]))
        return out / dE

    ## returns \f$ \sum_\alpha v^\alpha D^\alpha \f$ where \f$ D^\alpha_{ij} = d^\alpha_{ij} \f$
    def compute_NAC_matrix(self, velocity):
        nstates = self.nstates()
        out = np.zeros([nstates, nstates], dtype=np.complex64)
        ndim = self.ndim()
        assert(ndim == velocity.shape[0])
        for i in range(nstates):
            for j in range(i):
                dij = self.compute_derivative_coupling(i,j)
                out[i, j] = np.dot(velocity, dij)
                out[j, i] = - out[i, j]
        return out

## Class to propagate a single FSSH Trajectory
class Trajectory:
    def __init__(self, model, tracer, options):
        self.model = model
        self.tracer = tracer
        self.position = options["position"]
        self.velocity = options["velocity"]
        self.last_velocity = 0.0
        self.mass = options["mass"]
        if options["initial_state"] == "ground":
            self.rho = np.zeros([model.nstates(),model.nstates()], dtype = np.complex64)
            self.rho[0,0] = 1.0
            self.state = 0
        else:
            raise Exception("Unrecognized initial state option")

        # fixed initial parameters
        self.time = 0.0

        # read out of options
        self.dt = options["dt"]
        self.nsteps = int(options["total_time"] / self.dt)
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
        D = elec_states.compute_NAC_matrix(velo)

        nstates = self.model.nstates()

        G = np.zeros([nstates,nstates], dtype=np.complex64)
        for i in range(nstates):
            G[i,i] = elec_states.energies[i]

        G -= 1j * D

        if self.propagator == "exponential":
            diags, coeff = np.linalg.eigh(G)
            cmat = np.matrix(coeff)
            cmat_T = cmat.getH()
            cconj = np.array(cmat_T)
            tmp_rho = np.dot(cconj, np.dot(self.rho, coeff))
            nstates = model.nstates()
            for i in range(nstates):
                tmp_rho[i,:] *= np.exp(-1j * (diags[i] - diags[:]) * dt)
            self.rho[:] = np.dot(coeff, np.dot(tmp_rho, cconj))
        elif self.propagator == "ode":
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
        W = elec_states.compute_NAC_matrix(velo)

        probs = []

        for target in range(nstates):
            if (target == self.state):
                probs.append(0.0)
            else:
                bij = -2.0 * np.real(self.rho[self.state, target]) * W[target, self.state]

                # probability of hopping out of current state
                probs.append(self.dt * bij / np.real(self.rho[self.state, self.state]))

        P = sum(probs)
        accumulated_P = 0.0
        zeta = self.random.uniform()
        if zeta < P: # do switch
            for target in range(nstates):
                if (target == self.state): continue

                accumulated_P += probs[target]
                if (zeta < accumulated_P):
                    new_potential, old_potential = elec_states.energies[target], elec_states.energies[self.state]
                    delV = new_potential - old_potential
                    derivative_coupling = elec_states.compute_derivative_coupling(target, self.state)
                    component_kinetic = self.mode_kinetic_energy(derivative_coupling)
                    if delV <= component_kinetic:
                        self.state = target
                        self.rescale_component(derivative_coupling, -delV)
                        self.tracer.hops += 1
                break
        return P

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
        for step in range(self.nsteps):
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

            self.trace(electronics, prob, "collect")

        self.trace(electronics, prob, "finalize")

        return self.tracer

    ## Classifies end of simulation:
    #
    #  2*state + [0 for left, 1 for right]
    def outcome(self):
        out = np.zeros([2, self.model.nstates()])
        lr = 0 if self.position < 0.0 else 1
        if self.outcome_type == "populations":
            for ist in range(nstates):
                out[lr,ist] = np.real(self.rho[ist,ist])
        elif self.outcome_type == "state":
            out[lr,self.state] = 1.0
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

## Class to manage many FSSH trajectories
#
# Requires a model object which is a class that has functions V(x), dV(x), nstates(), and ndim()
# that return the Hamiltonian at position x, gradient of the Hamiltonian at position x
# number of electronic states, and dimension of nuclear space, respectively.
class FSSH:
    ## Constructor requires model and options input as kwargs
    # @param model object used to describe the model system
    #
    # Accepted keyword arguments and their defaults:
    # | key                |   default                  |
    # ---------------------|----------------------------|
    # | initial_state      | "ground"                   |
    # | position           |    -5.0                    |
    # | mass               | 2000.0                     |
    # | momentum           | 2.0                        |
    # | initial_time       | 0.0                        |
    # | dt                 | 0.1 / velocity            |
    # | total_time         | 2 * abs(position/velocity) |
    # | samples            | 2000                       |
    # | propagator         | "exponential"              |
    # | nprocs             | MultiProcessing.cpu_count  |
    # | outcome_type       | "state"                    |
    def __init__(self, model, tracemanager = TraceManager(), **inp):
        self.model = model
        self.tracemanager = tracemanager
        self.options = {}

        # system parameters
        self.options["initial_state"] = inp.get("initial_state", "ground")
        self.options["position"]      = inp.get("position", -5.0)
        self.options["mass"]          = inp.get("mass", 2000.0)
        self.options["velocity"]      = inp.get("momentum", 2.0) / self.options["mass"]

        # time parameters
        self.options["initial_time"]  = inp.get("initial_time", 0.0)
        self.options["dt"]            = inp.get("dt", 0.10 / self.options["velocity"])
        self.options["total_time"]    = inp.get("total_time", 2.0 * abs(self.options["position"] / self.options["velocity"]))

        # statistical parameters
        self.options["samples"]       = inp.get("samples", 2000)

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

    ## runs a set of trajectories and collects the results
    # @param n number of trajectories to run
    def run_trajectories(self, n):
        outcomes = np.zeros([2,self.model.nstates()])
        traces = []
        try:
            for it in range(n):
                traj = Trajectory(self.model, self.tracemanager.spawn_tracer(), self.options)
                trace = traj.simulate()
                traces.append(trace)
                outcomes += traj.outcome()
            return (outcomes, traces)
        except KeyboardInterrupt:
            pass

    ## runs many trajectories and returns averaged results
    def compute(self):
        # for now, define four possible outcomes of the simulation
        outcomes = np.zeros([2,self.model.nstates()])
        nsamples = int(self.options["samples"])
        energy_list = []
        nprocs = self.options["nprocs"]

        if nprocs > 1:
            pool = mp.Pool(nprocs)
            chunksize = (nsamples - 1)/nprocs + 1
            poolresult = []
            for ip in range(nprocs):
                batchsize = min(chunksize, nsamples - chunksize * ip)
                poolresult.append(pool.apply_async(unwrapped_run_trajectories, (self, batchsize)))
            try:
                for r in poolresult:
                    oc, tr = r.get(100)
                    outcomes += oc
                    self.tracemanager.add_batch(tr)
            except KeyboardInterrupt:
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

## global version of FSSH.run_trajectories that is necessary because of the stupid way threading pools work in python
def unwrapped_run_trajectories(fssh, n):
    return FSSH.run_trajectories(fssh, n)

if __name__ == "__main__":
    import tullymodels as tm
    import argparse as ap

    parser = ap.ArgumentParser(description="Example driver for FSSH")

    parser.add_argument('-m', '--model', default='simple', choices=[m for m in tm.modeldict], type=str, help="Tully model to plot (%(default)s)")
    parser.add_argument('-k', '--krange', default=(0.1,30.0), nargs=2, type=float, help="range of momenta to consider (%(default)s)")
    parser.add_argument('-n', '--nk', default=20, type=int, help="number of momenta to compute (%(default)d)")
    parser.add_argument('-s', '--samples', default=200, type=int, help="number of samples (%(default)d)")
    parser.add_argument('-j', '--nprocs', default=2, type=int, help="number of processors (%(default)d)")
    parser.add_argument('-p', '--propagator', default="exponential", choices=('exponential', 'ode'), type=str, help="propagator (%(default)s)")
    parser.add_argument('-M', '--mass', default=2000.0, type=float, help="particle mass (%(default)s)")
    parser.add_argument('-t', '--dt', default=None, type=float, help="time step (%(default)s)")
    parser.add_argument('-x', '--position', default=-5.0, type=float, help="starting position (%(default)s)")
    parser.add_argument('-o', '--output', default="averaged", type=str, help="what to print as output (%(default)s)")
    parser.add_argument('-z', '--seed', default=None, type=int, help="random seed (current date)")

    args = parser.parse_args()

    if args.model in tm.modeldict:
        model = tm.modeldict[args.model]()
    else:
        raise Exception("Unknown model chosen") # the argument parser should prevent this throw from being possible

    if (args.seed is not None):
        np.random.seed(args.seed)

    nk = args.nk
    min_k, max_k = args.krange

    kpoints = np.linspace(min_k, max_k, nk)
    for k in kpoints:
        fssh = FSSH(model, momentum = k,
                           position = args.position,
                           mass = args.mass,
                           samples = args.samples,
                           propagator = args.propagator,
                           nprocs = args.nprocs,
                           dt = args.dt,
                           seed = args.seed
                   )
        results = fssh.compute()
        outcomes = results.outcomes

        if (args.output == "averaged"):
            print "%12.6f %s" % (k, " ".join(["%12.6f" % x for x in np.nditer(outcomes, order='F')]))
        elif (args.output == "single"):
            for i in results.traces[0]:
                print "%12.6f %12.6f %12.6f %6d" % (i.time, i.position, i.momentum, i.activestate)
        else:
            print "Not printing results. This is probably not what you wanted!"
