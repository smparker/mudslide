#!/usr/bin/env python

import numpy as np
import scipy.integrate
import math as m
import multiprocessing as mp

## Tunneling through a single barrier model used in Tully's 1990 JCP
#
# \f[
#   V_{11} = \left\{ \begin{array}{cr}
#                   A (1 - e^{Bx}) & x < 0 \\
#                  -A (1 - e^{-Bx}) & x > 0
#                   \end{array} \right.
# \f]
# \f[ V_{22} = -V_{11} \f]
# \f[ V_{12} = V_{21} = C e^{-D x^2} \f]
class TullyModel:
    ## Constructor that defaults to the values reported in Tully's 1990 JCP
    def __init__(self, a = 0.01, b = 1.6, c = 0.005, d = 1.0):
        self.A = a
        self.B = b
        self.C = c
        self.D = d

    ## \f$V(x)\f$
    def V(self, x):
        v11 = m.copysign(self.A, x) * ( 1.0 - m.exp(-self.B * abs(x)) )
        v22 = -v11
        v12 = self.C * m.exp(-self.D * x * x)
        out = np.array([ [v11, v12],
                         [v12, v22] ])
        return out

    ## \f$\nabla V(x)\f$
    def Vgrad(self, R):
        v11 = self.A * self.B * m.exp(-self.B * abs(R))
        v22 = -v11
        v12 = -2.0 * self.C * self.D * R * m.exp(-self.D * R * R)
        out = np.array([ [v11, v12],
                         [v12, v22] ])
        return out

    def electronics(self, R, ref_coeff = None):
        return ElectronicStates(self.V(R), self.Vgrad(R), ref_coeff)

    def dim(self):
        return 2

## Wrapper around all the information computed for a set of electronics
#  states at a given position: V, dV, eigenvectors, eigenvalues
class ElectronicStates:
    ## Constructor
    # @param V Hamiltonian/potential
    # @param Vgrad Gradient of Hamiltonian/potential
    def __init__(self, V, Vgrad, ref_coeff = None):
        self.V = V
        self.dV = Vgrad
        self.energies, self.coeff = np.linalg.eigh(V)
        if ref_coeff is not None:
            for mo in range(self.dim()):
                if (np.dot(self.coeff[:,mo], ref_coeff[:,mo]) < 0.0):
                    self.coeff[:,mo] *= -1.0

    ## returns dimension of Hamiltonian
    def dim(self):
        return self.V.shape[0]

    ## returns \f$-\langle \phi_{\mbox{state}} | \nabla H | \phi_{\mbox{state}} \rangle\f$ of Hamiltonian
    # @param state state along which to get
    def compute_force(self, state):
        state_vec = self.coeff[:,state]
        force = - (np.dot(state_vec.T, np.dot(self.dV, state_vec)))
        return force

    def compute_potential(self, state):
        return self.energies[state]

    def compute_derivative_coupling(self, bra_state, ket_state):
        out = 0.0
        if (bra_state != ket_state):
            out = np.dot(self.coeff[:,bra_state].T, np.dot(self.dV, self.coeff[:,ket_state]))
            out /= self.energies[bra_state] - self.energies[ket_state]
        return out

    def compute_NAC_matrix(self, velocity):
        dim = self.dim()
        out = np.zeros([dim, dim], dtype=np.complex64)
        for i in range(dim):
            for j in range(i):
                out[i, j] = self.compute_derivative_coupling(i, j)
                out[j, i] = - out[i, j]
        out *= velocity
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
        G += 1j * D

        if self.propagator == "exponential":
            diags, coeff = np.linalg.eigh(G)
            cmat = np.matrix(coeff)
            cmat_T = cmat.getH()
            cconj = np.array(cmat_T)
            tmp_rho = np.dot(cconj, np.dot(self.rho, coeff))
            dim = model.dim()
            for i in range(dim):
                for j in range(dim):
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
        dij = elec_states.compute_derivative_coupling(self.state, target_state)
        bij = -2.0 * np.real(self.rho[self.state, target_state]) * np.dot(self.velocity, dij)
        #print "bij is %12.5f" % bij
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

    ## run simulation
    def simulate(self):
        electronics = model.electronics(self.position)
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
            new_electronics = model.electronics(self.position, electronics.coeff)
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
class FSSH:
    ## Constructor requires model and options input as kwargs
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

    def run_trajectories(self, n):
        outcomes = np.zeros([4])
        for it in range(n):
            traj = Trajectory(self.model, self.options)
            traj.simulate()
            outcomes += traj.outcome()
        return outcomes

    ## runs many trajectories and returns averaged results
    def compute(self):
        # for now, define four possible outcomes of the simulation
        outcomes = np.zeros([4])
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
            for r in poolresult:
                r.wait()
                outcomes += r.get(10)
            pool.close()
            pool.join()
        else:
            for it in range(nsamples):
                traj = Trajectory(self.model, self.options)
                energy_list.append(traj.simulate())
                outcomes += traj.outcome()

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

def unwrapped_run_trajectories(fssh, n):
    return FSSH.run_trajectories(fssh, n)

if __name__ == "__main__":
    model = TullyModel()

    nk = int(100)
    min_k = 4.0
    max_k = 30.0

    kpoints = np.linspace(min_k, max_k, nk)
    for k in kpoints:
        fssh = FSSH(model, momentum = k,
                           position = -5.0,
                           mass = 2000.0,
                           samples = 500,
                           propagator = "exponential",
                           nprocs = 4
                   )
        results = fssh.compute()
        print "%12.6f %12.6f %12.6f %12.6f %12.6f" % (k, results[0], results[1], results[2], results[3])
