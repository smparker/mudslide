#!/usr/bin/env python

import numpy as np
import scipy.integrate
import math as m

## Tunneling through a single barrier model used in Tully's 1990 JCP
class TullyModel:
    # parameters in Tully's model
    A = 0.0
    B = 0.0
    C = 0.0
    D = 0.0

    # default to the values reported in Tully's 1990 JCP
    def __init__(self, a = 0.01, b = 1.6, c = 0.005, d = 1.0):
        self.A = a
        self.B = b
        self.C = c
        self.D = d

    def V(self, R):
        v11 = m.copysign(self.A, R) * ( 1.0 - m.exp(-self.B * abs(R)) )
        v22 = -v11
        v12 = self.C * m.exp(-self.D * R * R)
        out = np.array([ [v11, v12],
                         [v12, v22] ])
        return out

    def Vgrad(self, R):
        v11 = self.A * self.B * m.exp(-self.B * abs(R))
        v22 = -v11
        v12 = -2.0 * self.C * self.D * R * m.exp(-self.D * R * R)
        out = np.array([ [v11, v12],
                         [v12, v22] ])
        return out

## Wrapper around all the information computed for a set of electronics
#  states at a given position: V, dV, eigenvectors, eigenvalues
class ElectronicStates:
    ## \f$H\f$
    V = np.zeros([2, 2])
    ## \f$\nabla H\f$
    dV = np.zeros([2, 2])
    ## eigenvectors of \f$H\f$
    coeff = np.zeros([2,2])
    ## eigenvalues of \f$H\f$
    energies = np.zeros([2])

    ## Constructor
    # @param V Hamiltonian/potential
    # @param Vgrad Gradient of Hamiltonian/potential
    def __init__(self, V, Vgrad):
        self.V = V
        self.dV = Vgrad
        self.energies, self.coeff = np.linalg.eigh(V)

    ## returns dimension of Hamiltonian
    def dim(self):
        return self.V.shape[0]

    ## returns \f$-\langle \phi_{\mbox{state}} | \nabla H | \phi_{\mbox{state}} \rangle\f$ of Hamiltonian
    # @param state state along which to get
    def compute_force(self, state):
        state_vec = self.coeff[:,state]
        force = - (np.dot(state_vec.T, np.dot(self.dV, state_vec)))
        return force

    def compute_derivative_coupling(self, bra_state, ket_state):
        out = 0.0
        if (bra_state != ket_state):
            out = np.dot(self.coeff[:,bra_state].T, np.dot(self.dV, self.coeff[:,ket_state]))
            out /= self.energies[bra_state] - self.energies[ket_state]
        return out

    def compute_NAC_matrix(self, velocity):
        out = np.zeros([self.dim(), self.dim()], dtype=np.complex64)
        out[0,1] = self.compute_derivative_coupling(0, 1)
        out[1,0] = - out[0,1]
        out *= -1j * velocity
        return out

## Class to propagate a single FSSH Trajectory
class Trajectory:
    model = TullyModel()
    position = 0.0
    velocity = 0.0
    acceleration = 0.0
    time = 0.0
    mass = 1.0
    rho = np.zeros([2,2], dtype=np.complex64, order='F')
    step = 0
    state = 0
    dt = 0.1

    def __init__(self, model, options):
        # input explicitly
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
        self.step = 0

        # read out of options
        self.dt = options["dt"]
        self.nsteps = int(options["total_time"] / self.dt)

    def kinetic_energy(self, component = None):
        if component == None:
            component = self.velocity
        return 0.5 * self.mass * np.dot(component, component)

    def mode_kinetic_energy(self, direction):
        component = np.dot(direction, self.velocity) / np.dot(direction, direction) * direction
        return kinetic_energy(self, component)

    ## Rescales velocity in the specified direction and amount
    # @param direction array specifying the direction of the velocity to rescale
    # @param reduction scalar specifying how much kinetic energy should be damped
    def rescale_component(self, direction, reduction):
        # normalize
        direction /= np.dot(direction, direction)
        Md = self.mass * direction
        a = np.dot(Md, Md)
        b = 2.0 * self.mass * np.dot(self.velocity, Md)
        c = reduction
        roots = np.roots([a, b, c])
        scal = min(roots, key=lambda x: abs(x))
        self.velocity += scale * direction

    def step_forward(self):
        self.position += self.velocity * self.dt + 0.5 * self.acceleration * self.dt * self.dt

    ## Propagates \f$\rho(t)\f$ to \f$\rho(t + dt)\f$
    # @param elec_states_0 ElectronicStates at \f$t\f$
    # @param elec_states_1 ElectronicStates at \f$t + dt\f$
    def propagate_rho(self, elec_states_0, elec_states_1):
        def drho(time, y):
            ymat = np.reshape(y, [2, 2])
            H = 0.5 * (elec_states_0.V + elec_states_1.V)
            D = 0.5 * (elec_states_0.compute_NAC_matrix(self.velocity)
                        + elec_states_1.compute_NAC_matrix(self.velocity))
            tmp = np.dot(H + D, ymat)
            dro = -1j * (tmp - np.conj(tmp.T))
            return np.reshape(dro, [4])

        rhovec = np.reshape(self.rho, [4])
        integrator = scipy.integrate.complex_ode(drho).set_integrator('vode', method='bdf', with_jacobian=False)
        integrator.set_initial_value(rhovec, self.time)
        integrator.integrate(self.time + self.dt)
        self.rho = integrator.y
        if not integrator.successful():
            exit("Propagation of the electronic wavefunction failed!")

    def surface_hopping(self, elec_states):
        d01 = elec_states.compute_derivative_coupling(0, 1)
        b01 = -2.0 * np.real(self.rho[0,1]) * self.velocity * d01
        # probability of hopping out of current state
        P = self.dt * b01 / self.rho[self.state, self.state]
        zeta = np.random.uniform()
        if zeta < P: # do switch
            # beware, this will only work for a two-state model
            target_state = 1-self.state
            new_potential, old_potential = elec_states.energies[target_state], elec_states.energies[self.state]
            delV = new_potential - old_potential
            component_kinetic = mode_kinetic_energy(np.ones([1]))
            if delV <= component_kinetic:
                self.state = target_state
                rescale_component(np.ones([1]), delV)

    def simulate(self):
        electronics = ElectronicStates(model.V(self.position), model.Vgrad(self.position))
        # start by taking half step in velocity
        initial_acc = electronics.compute_force(self.state) / self.mass
        self.velocity += 0.5 * initial_acc * self.dt

        # propagation
        for step in range(self.nsteps):
            # first update nuclear coordinates
            self.position += self.velocity * self.dt
            # calculate electronics at new position
            new_electronics = ElectronicStates(model.V(self.position), model.Vgrad(self.position))
            acceleration = new_electronics.compute_force(self.state) / self.mass
            self.velocity += acceleration * self.dt

            # now propagate the electronic wavefunction to the new time
            self.propagate_rho(electronics, new_electronics)

    def outcome(self):
        # first bit is left (0) or right (1), second bit is electronic state
        lr = 0 if self.position < 0.0 else 1
        return 2*self.state + lr

## Class to manage many FSSH trajectories
class FSSH:
    ## model to be used for the simulations
    model = TullyModel()
    ## input options
    options = {}

    def __init__(self, model, **inp):
        self.model = model

        # system parameters
        self.options["initial_state"] = inp.get("initial_state", "ground")
        self.options["position"]      = inp.get("position", -10.0)
        self.options["mass"]          = inp.get("mass", 2000.0)
        self.options["velocity"]      = inp.get("momentum", 2.0) / self.options["mass"]

        # time parameters
        self.options["initial_time"]  = inp.get("initial_time", 0.0)
        self.options["dt"]            = inp.get("dt", 0.1)
        self.options["total_time"]    = inp.get("total_time", abs(self.options["position"] / self.options["velocity"]))

        # statistical parameters
        self.options["samples"]       = inp.get("samples", 2000)

    def compute(self):
        # for now, define four possible outcomes of the simulation
        outcomes = np.zeros([4])
        nsamples = int(self.options["samples"])
        for it in range(nsamples):
            traj = Trajectory(self.model, self.options)
            traj.simulate()
            outcomes[traj.outcome()] += 1
        outcomes /= float(nsamples)
        return outcomes

if __name__ == "__main__":
    model = TullyModel()

    nk = int(100)
    max_k = float(30.0)
    min_k = max_k / nk
    kpoints = np.linspace(min_k+5, max_k, nk)
    for k in kpoints:
        fssh = FSSH(model, momentum = k,
                           position = -5.0,
                           mass = 2000.0,
                           total_time = 10.0 / (k / 2000.0),
                           dt = 50,
                           samples = 100)
        results = fssh.compute()
        print "%12.6f %12.6f %12.6f %12.6f %12.6f" % (k, results[0], results[1], results[2], results[3])
