#!/usr/bin/env python

import numpy as np
import math as m

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

#class FSSH:
#    

class ElectronicStates:
    """ Wrapper around all the information computed for a set of electronics
        states at a given position: V, dV, eigenvectors, eigenvalues """
    V = np.zeros([2, 2])
    dV = np.zeros([2, 2])
    coeff = np.zeros([2,2])
    energies = np.zeros([2])

    def __init__(self, V, Vgrad):
        self.V = V
        self.dV = Vgrad
        self.energies, self.coeff = np.linalg.eigh(V)

    def dim(self):
        return V.shape[0]

    def compute_force(self, state):
        state_vec = coeff[:,state]
        force = - (np.dot(state_vec.T, np.dot(dV, state_vec)))
        return force

    def compute_derivative_coupling(self, bra_state, ket_state):
        out = 0.0
        if (bra_state != ket_state):
            out = np.dot(coeff[:,bra_state].T, np.dot(dV, coeff[:,ket_state]))
            out /= energies[bra_state] - energies[ket_state]
        return out

    def compute_NAC_matrix(self, velocity):
        out = np.zeros([dim(), dim()], dtype=complex64)
        out[0,1] = compute_derivative_coupling(0, 1)
        out[1,0] = - out[0,1]
        out *= -1j * velocity
        return out

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

    def __init__(self, mod, p, v, m, initial_rho, initial_state, options):
        # input explicitly
        self.model = mod
        self.position = p
        self.velocity = v
        self.mass = m
        self.rho = initial_rho
        self.state = initial_state

        # fixed initial parameters
        self.time = 0.0
        self.step = 0

        # read out of options
        self.dt = options["dt"]
        self.nsteps = options["nsteps"]

    def kinetic_energy(self, component = self.velocity):
        return 0.5 * self.mass * np.dot(component, component)

    def mode_kinetic_energy(self, direction):
        component = np.dot(direction, self.velocity) / np.dot(direction, direction) * direction
        return kinetic_energy(component)

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

    def compute_force(self, phi, Vgrad):

    ## Propagates \f$\rho(t)\f$ to \f$\rho(t + dt)\f$
    # @param elec_states_0 ElectronicStates at \f$t\f$
    # @param elec_states_1 ElectronicStates at \f$t + dt\f$
    def propagate_rho(self, elec_states_0, elec_states_1):
        def drho(time, y):
            H = 0.5 * (elec_states_0.V + elec_states_1.V)
            D = 0.5 * (elec_states_0.compute_NAC_matrix(self.velocity)
                        + elec_states_1.compute_NAC_matrix(self.velocity))
            tmp = np.dot(H + D, y)
            dro = -1j * (tmp - np.conj(tmp.T))
            return dro

        integrator = scipy.integrate.ode(drho).set_integrator('zvode', method='bdf', with_jacobian=False)
        integrator.set_initial_value(self.rho, self.time)
        integrator.integrate(self.time + self.dt)
        self.rho = integrator.y
        if !integrator.successful():
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
            component_kinetic = mode_kinetic_energy(np.ones([1])
            if delV <= component_kinetic:
                self.state = target_state
                rescale_component(np.ones([1]), delV)

    def simulate(self):
        electronics = ElectronicState(model.V(self.position), model.Vgrad(self.position))
        # start by taking half step in velocity
        initial_acc = electronics.compute_force(self.state) / self.mass
        self.velocity += 0.5 * initial_acc * self.dt

        # propagation
        for step in range(nsteps):
            # first update nuclear coordinates
            self.position += self.velocity * self.dt
            # calculate electronics at new position
            new_electronics = ElectronicState(model.V(self.position), model.Vgrad(self.position))
            acceleration = new_electronics.compute_force(self.state) / self.mass
            self.velocity += acceleration * self.dt

            # now propagate the electronic wavefunction to the new time
            propagate_rho(electronics, new_electronics)

model = TullyModel()

# compute barrier

diss_ham = model.V(-10000)
diss_energy = np.linalg.eigvalsh(diss_ham)[0]
int_ham = model.V(0)
int_energy = np.linalg.eigvalsh(int_ham)[0]

# print surfaces for debugging
nr = np.linspace(-10, 10, 200 + 1)

traj = Trajectory(-13, 0.10000, 1, np.zeros([2,2]), {"dt" : 0.1})
nsteps = 5000
for i in range(nsteps):
    ham = model.V(traj.position)
    dham = model.Vgrad(traj.position)
    energies, adiabats = np.linalg.eigh(ham)
    traj.nuclear_step(adiabats[:,0], dham)
    print "%6d %12.6f" % (i, traj.position)

#
#for r in nr:
#    ham = model.V(r)
#    dham = model.Vgrad(r)
#    energies, adiabats = np.linalg.eigh(ham)
#    force = traj.nuclear_step(adiabats[:,0], dham)
#    d = traj.derivative_coupling(adiabats[:,0], energies[0], adiabats[:,1], energies[1], dham)
#    #print "%12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f" % (r, ham[0,0], ham[1,1], energies[0], energies[1], gradient, d)
#    print "%12.6f %12.6f %12.6f %12.6f %12.6f %12.6f" % (r, ham[0,0], energies[0], dham[0,0], force, d)
