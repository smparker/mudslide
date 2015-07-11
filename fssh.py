## @package fssh
#  Module responsible for propagating surface hopping trajectories

import numpy as np
import scipy.integrate
import math as m
import multiprocessing as mp

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
        return self.V.shape[0]

    ## returns dimensionality of model (nuclear coordinates)
    def ndim(self):
        return self.dV.shape[2]

    ## returns \f$-\langle \phi_{\mbox{state}} | \nabla H | \phi_{\mbox{state}} \rangle\f$ of Hamiltonian
    # @param state state along which to compute force
    def compute_force(self, state):
        out = np.zeros(self.ndim())
        state_vec = self.coeff[:,state]
        for d in range(self.ndim()):
            out[d] = - (np.dot(state_vec.T, np.dot(self.dV[:,:,d], state_vec)))
        return out

    ## returns \f$\phi_{\mbox{state}} | H | \phi_{\mbox{state}} = \varepsilon_{\mbox{state}}\f$
    def compute_potential(self, state):
        return self.energies[state]

    ## returns \f$\phi_{i} | \nabla_\alpha \phi_{j} = d^\alpha_{ij}\f$
    def compute_derivative_coupling(self, bra_state, ket_state):
        out = np.zeros(self.ndim())
        if (bra_state != ket_state):
            for d in range(self.ndim()):
                dij = np.dot(self.coeff[:,bra_state].T, np.dot(self.dV[:,:,d], self.coeff[:,ket_state]))
                dE = self.energies[ket_state] - self.energies[bra_state]
                if abs(dE) < 1.0e-14:
                    dE = m.copysign(1.0e-14, dE)
                out[d] = dij / dE
        return out

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
    def __init__(self, model, options):
        self.model = model
        self.position = options["position"]
        self.velocity = options["velocity"]
        self.mass = options["mass"]
        if options["initial_state"] == "ground":
            self.rho = np.zeros([2,2], dtype = np.complex64)
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
        direction /= m.sqrt(np.dot(direction, direction))
        Md = self.mass * direction
        a = np.dot(Md, Md)
        b = 2.0 * np.dot(self.mass * self.velocity, Md)
        c = -2.0 * self.mass * reduction
        roots = np.roots([a, b, c])
        scal = min(roots, key=lambda x: abs(x))
        self.velocity += scal * direction

    ## Propagates \f$\rho(t)\f$ to \f$\rho(t + dt)\f$
    # @param elec_states_0 ElectronicStates at \f$t\f$
    # @param elec_states_1 ElectronicStates at \f$t + dt\f$
    #
    # The propagation assumes the electronic energies and couplings are static throughout.
    # This will only be true for fairly small time steps
    def propagate_rho(self, elec_states_0, elec_states_1):
        D = elec_states_0.compute_NAC_matrix(self.velocity)

        G = np.zeros([2,2], dtype=np.complex64)
        G[0,0] = elec_states_0.energies[0]
        G[1,1] = elec_states_0.energies[1]
        G -= 1j * D

        if self.propagator == "exponential":
            diags, coeff = np.linalg.eigh(G)
            cmat = np.matrix(coeff)
            cmat_T = cmat.getH()
            cconj = np.array(cmat_T)
            tmp_rho = np.dot(cconj, np.dot(self.rho, coeff))
            nstates = model.nstates()
            for i in range(nstates):
                for j in range(nstates):
                    tmp_rho[i,j] *= np.exp(-1j * (diags[i] - diags[j]) * self.dt)
            self.rho[:] = np.dot(coeff, np.dot(tmp_rho, cconj))
        elif self.propagator == "ode":
            G *= -1j
            def drho(time, y):
                ymat = np.reshape(y, [2, 2])
                dro = np.dot(G, ymat) - np.dot(ymat, G)
                return np.reshape(dro, [4])

            rhovec = np.reshape(self.rho, [4])
            integrator = scipy.integrate.complex_ode(drho).set_integrator('vode', method='bdf', with_jacobian=False)
            integrator.set_initial_value(rhovec, self.time)
            integrator.integrate(self.time + self.dt)
            self.rho = np.reshape(integrator.y, [2,2])
            if not integrator.successful():
                exit("Propagation of the electronic wavefunction failed!")
        else:
            raise Exception("Unrecognized method for propagation of electronic density matrix!")

    ## Compute probability of hopping, generate random number, and perform hops
    def surface_hopping(self, elec_states):
        # this trick is only valid for 2 state problem
        target_state = 1-self.state
        dij = elec_states.compute_derivative_coupling(target_state, self.state)
        bij = -2.0 * np.real(self.rho[self.state, target_state]) * np.dot(self.velocity, dij)
        # probability of hopping out of current state
        P = self.dt * bij / np.real(self.rho[self.state, self.state])
        zeta = np.random.uniform()
        if zeta < P: # do switch
            # beware, this will only work for a two-state model
            new_potential, old_potential = elec_states.energies[target_state], elec_states.energies[self.state]
            delV = new_potential - old_potential
            component_kinetic = self.mode_kinetic_energy(np.ones([1]))
            if delV <= component_kinetic:
                self.state = target_state
                self.rescale_component(np.ones([1]), -delV)

    ## helper function to simplify the calculation of the electronic states at a given position
    def compute_electronics(self, position, ref_coeff = None):
        return ElectronicStates(model.V(position), model.dV(position), ref_coeff)

    ## run simulation
    def simulate(self):
        electronics = self.compute_electronics(self.position)
        # start by taking half step in velocity
        initial_acc = electronics.compute_force(self.state) / self.mass
        self.velocity += 0.5 * initial_acc * self.dt
        potential_energy = electronics.compute_potential(self.state)
        energy_list = [ (potential_energy, self.total_energy(electronics)) ]

        # propagation
        for step in range(self.nsteps):
            # first update nuclear coordinates
            self.position += self.velocity * self.dt
            # calculate electronics at new position
            new_electronics = self.compute_electronics(self.position, electronics.coeff)
            acceleration = new_electronics.compute_force(self.state) / self.mass
            self.velocity += acceleration * self.dt

            # now propagate the electronic wavefunction to the new time
            self.propagate_rho(electronics, new_electronics)
            self.surface_hopping(new_electronics)
            electronics = new_electronics
            self.time += self.dt
            energy_list.append((electronics.compute_potential(self.state), self.total_energy(electronics)))
        return energy_list

    ## Classifies end of simulation:
    #
    # result | classification
    # -------|---------------
    #   0    | lower state on the left
    #   1    | lower state on the right
    #   2    | upper state on the left
    #   3    | upper state on the right
    def outcome(self):
        out = np.zeros([4])
        lr = 0 if self.position < 0.0 else 1
        if self.outcome_type == "populations":
            out[lr] = np.real(self.rho[0,0])
            out[2 + lr] = np.real(self.rho[1,1])
        elif self.outcome_type == "state":
            out[2*self.state + lr] = 1.0
        else:
            raise Exception("Unrecognized outcome recognition type")
        return out
        # first bit is left (0) or right (1), second bit is electronic state

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
    # | dt                 | 0.05 / velocity            |
    # | total_time         | 2 * abs(position/velocity) |
    # | samples            | 2000                       |
    # | propagator         | "exponential"              |
    # | nprocs             | MultiProcessing.cpu_count  |
    # | outcome_type       | "state"                    |
    def __init__(self, model, **inp):
        self.model = model
        self.options = {}

        # system parameters
        self.options["initial_state"] = inp.get("initial_state", "ground")
        self.options["position"]      = inp.get("position", -5.0)
        self.options["mass"]          = inp.get("mass", 2000.0)
        self.options["velocity"]      = inp.get("momentum", 2.0) / self.options["mass"]

        # time parameters
        self.options["initial_time"]  = inp.get("initial_time", 0.0)
        self.options["dt"]            = inp.get("dt", 0.05 / self.options["velocity"])
        self.options["total_time"]    = inp.get("total_time", 2.0 * abs(self.options["position"] / self.options["velocity"]))

        # statistical parameters
        self.options["samples"]       = inp.get("samples", 2000)

        # numerical parameters
        self.options["propagator"]    = inp.get("propagator", "exponential")
        if self.options["propagator"] not in ["exponential", "ode"]:
            raise Exception("Unrecognized electronic propagator!")
        self.options["nprocs"]        = inp.get("nprocs", mp.cpu_count())
        self.options["outcome_type"]  = inp.get("outcome_type", "state")

    ## runs a set of trajectories and collects the results
    # @param n number of trajectories to run
    def run_trajectories(self, n):
        outcomes = np.zeros([4])
        en_list = []
        try:
            for it in range(n):
                traj = Trajectory(self.model, self.options)
                potentials = traj.simulate()
                en_list.append(potentials)
                outcomes += traj.outcome()
            return (outcomes, en_list)
        except KeyboardInterrupt:
            pass

    ## runs many trajectories and returns averaged results
    def compute(self):
        # for now, define four possible outcomes of the simulation
        outcomes = np.zeros([4])
        nsamples = int(self.options["samples"])
        energy_list = []
        nprocs = self.options["nprocs"]

        pool = mp.Pool(nprocs)
        chunksize = (nsamples - 1)/nprocs + 1
        poolresult = []
        for ip in range(nprocs):
            batchsize = min(chunksize, nsamples - chunksize * ip)
            poolresult.append(pool.apply_async(unwrapped_run_trajectories, (self, batchsize)))
        try:
            for r in poolresult:
                oc, en = r.get(100)
                outcomes += oc
                energy_list.extend(en)
        except KeyboardInterrupt:
                exit(" Aborting!")
        pool.close()
        pool.join()

        #nsteps = len(energy_list[0])
        #t = self.options["initial_time"]
        #for i in range(nsteps):
        #    out = "%12.6f" % t
        #    for j in range(nsamples):
        #        out += " %12.6f %12.6f" % energy_list[j][i]
        #    print out
        #    t += self.options["dt"]

        outcomes /= float(nsamples)
        return outcomes

## global version of FSSH.run_trajectories that is necessary because of the stupid way threading pools work in python
def unwrapped_run_trajectories(fssh, n):
    return FSSH.run_trajectories(fssh, n)

if __name__ == "__main__":
    import tullymodels as tully

    model = tully.TullySimpleAvoidedCrossing()

    nk = int(20)
    min_k = 0.1
    max_k = 30.0

    kpoints = np.linspace(min_k, max_k, nk)
    for k in kpoints:
        fssh = FSSH(model, momentum = k,
                           position = -10.0,
                           mass = 2000.0,
                           samples = 2000,
                           propagator = "exponential",
                           nprocs = 4
                   )
        results = fssh.compute()
        print "%12.6f %12.6f %12.6f %12.6f %12.6f" % (k, results[0], results[1], results[2], results[3])
