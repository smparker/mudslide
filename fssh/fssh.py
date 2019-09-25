#!/usr/bin/env python
## @package fssh
#  Module responsible for propagating surface hopping trajectories

# fssh: program to run surface hopping simulations for model problems
# Copyright (C) 2018, Shane Parker <smparker@uci.edu>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
    # @param reference [optional] set of coefficients (for example, from a previous step) used to keep the sign of eigenvectors consistent
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
    # @param reference optional ElectronicStates from previous step used only to fix phase
    def _compute_coeffs(self, reference=None):
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
        nst = self.nstates()
        ndim = self.ndim()

        half = np.einsum("xij,jp->ipx", self._dV, self._coeff)

        out = np.zeros([nst, ndim])
        for ist in range(self.nstates()):
            out[ist,:] += -np.einsum("i,ix->x", self._coeff[:,ist], half[:,ist,:])
        return out

    ## returns \f$\phi_{\mbox{state}} | H | \phi_{\mbox{state}} = \varepsilon_{\mbox{state}}\f$
    def _compute_hamiltonian(self):
        return np.dot(self._coeff.T, np.dot(self._V, self._coeff))

    ## returns \f$\phi_{i} | \nabla_\alpha \phi_{j} = d^\alpha_{ij}\f$
    def _compute_derivative_coupling(self):
        out = np.einsum("ip,xij,jq->pqx", self._coeff, self._dV, self._coeff)

        for j in range(self.nstates()):
            for i in range(j):
                dE = self._energies[j] - self._energies[i]
                if abs(dE) < 1.0e-14:
                    dE = m.copysign(1.0e-14, dE)

                out[i,j,:] /= dE
                out[j,i,:] /= -dE

        return out

    ## returns \f$ \sum_\alpha v^\alpha D^\alpha \f$ where \f$ D^\alpha_{ij} = d^\alpha_{ij} \f$
    # @param velocity [ndim] numpy array of velocities
    def NAC_matrix(self, velocity):
        out = np.einsum("pqx,x->pq", self.derivative_coupling, velocity)
        return out

    ## returns \f$F^\xi{ij} = \langle \phi_i | -\nabla_\xi H | \phi_j\rangle\f$
    def force_matrix(self):
        out = -np.einsum("ip,xij,jq->pqx", self._coeff, self._dV, self._coeff)
        return out

## Class to propagate a single SH Trajectory
class TrajectorySH(object):
    ## Constructor
    # @param model Model object defining problem
    # @param tracer spawn from TraceManager to collect results
    # @param options option dictionary
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

    ## Return random number for hopping decisions
    def random(self):
        return self.random_state.uniform()

    ## Is trajectory still inside interaction region?
    def currently_interacting(self):
        """determines whether trajectory is currently inside an interaction region"""
        return self.box_bounds[0] < self.position and self.box_bounds[1] > self.position

    ## Initializes variables related to continue_simulating
    def duration_initialize(self):
        """Initializes variables related to continue_simulating"""
        self.found_box = False
        self.box_bounds = (-5,5)
        self.max_steps = 10000

    ## Returns True if a trajectory ought to keep running, False if it should finish
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

    ## add results from current time point to tracing function TODO: ugly
    def trace(self, electronics, prob, call):
        if call == "collect":
            func = self.tracer.collect
        elif call == "finalize":
            func = self.tracer.finalize

        func(self.time, np.copy(self.position), self.mass*np.copy(self.velocity),
            self.potential_energy(electronics), self.kinetic_energy(), self.total_energy(electronics),
            np.copy(self.rho), self.state, electronics, prob)

    ## current kinetic energy
    def kinetic_energy(self):
        return 0.5 * self.mass * np.dot(self.velocity, self.velocity)

    ## current potential energy
    # @param electronics ElectronicStates from current step
    def potential_energy(self, electronics):
        return electronics.hamiltonian[self.state,self.state]

    ## current kinetic + potential energy
    # @param electronics ElectronicStates from current step
    def total_energy(self, electronics):
        potential = self.potential_energy(electronics)
        kinetic = self.kinetic_energy()
        return potential + kinetic

    ## force on active state
    # @param electronics ElectronicStates from current step
    def force(self, electronics):
        return electronics.force[self.state,:]

    ## kinetic energy along given mode
    # @param direction [ndim] numpy array defining direction
    def mode_kinetic_energy(self, direction):
        component = np.dot(direction, self.velocity) / np.dot(direction, direction) * direction
        return 0.5 * self.mass * np.dot(component, component)

    ## Return direction in which to rescale momentum
    # @param tau derivative coupling vector
    # @param source active state before hop
    # @param target active state after hop
    def rescale_direction(self, tau, source, target):
        return tau

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

    ## Compute the Hamiltonian used to propagate the electronic wavefunction
    #  returns nonadiabatic coupling H - i W
    #  @param elec_states ElectronicStates at \f$t\f$
    def hamiltonian_propagator(self, elec_states):
        velo = 0.5*(self.last_velocity + self.velocity)

        out = elec_states.hamiltonian - 1j * elec_states.NAC_matrix(velo)
        return out

    ## Propagates \f$\rho(t)\f$ to \f$\rho(t + dt)\f$
    # @param elec_states ElectronicStates at \f$t\f$
    # @param dt time step
    #
    # The propagation assumes the electronic energies and couplings are static throughout.
    # This will only be true for fairly small time steps
    def propagate_electronics(self, elec_states, dt):
        W = self.hamiltonian_propagator(elec_states)

        diags, coeff = np.linalg.eigh(W)

        # use W as temporary storage
        U = np.linalg.multi_dot([ coeff, np.diag(np.exp(-1j * diags * dt)), coeff.T.conj() ])
        np.dot(U, np.dot(self.rho, U.T.conj(), out=W), out=self.rho)

    ## move classical position forward one step
    # @param electronics ElectronicStates from current step
    def advance_position(self, electronics):
        self.position += self.velocity * self.dt

    ## move classical velocity forward one step
    # @param electronics ElectronicStates from current step
    def advance_velocity(self, electronics):
        acceleration = self.force(electronics) / self.mass

        self.last_velocity, self.velocity = self.velocity, self.velocity + acceleration * self.dt

    ## Compute probability of hopping, generate random number, and perform hops
    # @param elec_states ElectronicStates from current step
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
                u = self.rescale_direction(derivative_coupling, self.state, hop_to)
                self.rescale_component(u, -delV)
                self.tracer.hops += 1

        return sum(probs)

    ## given a set of probabilities, determines whether and where to hop
    # @param probs [nstates] numpy array of individual hopping probabilities
    #  returns (do_hop, target_state)
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
    # @param position classical positions
    # @param electronics ElectronicStates from previous step
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
        self.propagate_electronics(electronics, 0.5*self.dt)
        potential_energy = self.potential_energy(electronics)

        prob = 0.0

        self.trace(electronics, prob, "collect")

        # propagation
        while (True):
            # first update nuclear coordinates
            self.advance_position(electronics)

            # calculate electronics at new position
            last_electronics, electronics = electronics, self.compute_electronics(self.position, electronics)

            # update velocity
            self.advance_velocity(electronics)

            # now propagate the electronic wavefunction to the new time
            self.propagate_electronics(electronics, self.dt)
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
TraceData = collections.namedtuple('TraceData', 'time position momentum potential kinetic energy rho activestate electronics hopping')

## Collect results from a single trajectory
class Trace(object):
    def __init__(self):
        self.data = []
        self.hops = 0

    ## collect and optionally process data
    def collect(self, time, position, momentum, potential_energy, kinetic_energy, total_energy, rho, activestate, electronics, prob):
        self.data.append(TraceData(time=time, position=position, momentum=momentum, potential=potential_energy, kinetic=kinetic_energy,
                            energy=total_energy, rho=rho, activestate=activestate, electronics=electronics, hopping=prob))

    ## finalize an individual trajectory
    def finalize(self, time, position, momentum, potential_energy, kinetic_energy, total_energy, rho, activestate, electronics, prob):
        self.collect(time, position, momentum, potential_energy, kinetic_energy, total_energy, rho, activestate, electronics, prob)

    def __iter__(self):
        return self.data.__iter__()

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)

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

## Exception class indicating that a simulation was terminated while still inside the "interaction region"
class StillInteracting(Exception):
    def __init__(self):
        Exception.__init__(self, "A simulation ended while still inside the interaction region.")


## Trajectory surface hopping using a cumulative approach rather than instantaneous
#
#  Instead of using a random number generator at every time step to test for a hop,
#  hops occur when the cumulative probability of a hop crosses a randomly determined
#  threshold. Swarmed results should be identical to the traditional variety, but
#  should be a bit easier to reproduce since far fewer random numbers are ever needed.
class TrajectoryCum(TrajectorySH):
    ## Constructor (see TrajectorySH constructor)
    def __init__(self, *args, **kwargs):
        TrajectorySH.__init__(self, *args, **kwargs)

        self.prob_cum = 0.0
        self.zeta = self.random()

    ## given a set of probabilities, determines whether and where to hop
    # @param probs [nstates] numpy array of individual hopping probabilities
    #  returns (do_hop, target_state)
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

## Ehrenfest dynamics
class Ehrenfest(TrajectorySH):
    def __init__(self, *args, **kwargs):
        TrajectorySH.__init__(self, *args, **kwargs)

    ## Ehrenfest potential energy = tr(rho * H)
    # @param electronics ElectronicStates from current step
    def potential_energy(self, electronics):
        return np.real(np.trace(np.dot(self.rho, electronics.hamiltonian)))

    ## Ehrenfest force = tr(rho * H')
    # @param electronics ElectronicStates from current step
    def force(self, electronics):
        return np.dot(np.real(np.diag(self.rho)), electronics.force)

    ## Ehrenfest never hops
    # @param electronics ElectronicStates from current step (not used)
    def surface_hopping(self, electronics):
        return 0.0

