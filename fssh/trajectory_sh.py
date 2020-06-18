#!/usr/bin/env python
## @package trajectorysh
#  Module responsible for propagating surface hopping trajectories

# fssh: program to run surface hopping simulations for model problems
# Copyright (C) 2018-2020, Shane Parker <shane.parker@case.edu>
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

from .version import __version__
from .propagation import rk4

import copy as cp
import numpy as np
import sys

## Class to propagate a single SH Trajectory
class TrajectorySH(object):
    ## Constructor
    # @param model Model object defining problem
    # @param tracer spawn from TraceManager to collect results
    # @param options option dictionary
    def __init__(self, model, x0, p0, rho0, tracer=None, queue=None, **options):
        self.model = model
        self.tracer = tracer if tracer is not None else Trace()
        self.queue = queue
        self.mass = model.mass
        self.position = np.array(x0, dtype=np.float64).reshape(model.ndim())
        self.velocity = np.array(p0, dtype=np.float64).reshape(model.ndim()) / self.mass
        self.last_velocity = np.zeros_like(self.velocity, dtype=np.float64)
        if "last_velocity" in options:
            self.last_velocity[:] = options["last_velocity"]
        if np.isscalar(rho0):
            if rho0 == "ground":
                self.rho = np.zeros([model.nstates(),model.nstates()], dtype=np.complex128)
                self.rho[0,0] = 1.0
                self.state = 0
            else:
                Exception("Unrecognized initial state option")
        else:
            try:
                self.rho = np.copy(rho0)
                self.state = int(options["state0"])
            except:
                raise Exception("Unrecognized initial state option")

        # function duration_initialize should get us ready to for future continue_simulating calls
        # that decide whether the simulation has finished
        if "duration" in options:
            self.duration = options["duration"]
        else:
            self.duration_initialize(options)

        # fixed initial parameters
        self.time = float(options.get("t0", 0.0))
        self.nsteps = int(options.get("previous_steps", 0))
        self.trace_every = int(options.get("trace_every", 1))

        # read out of options
        self.dt = float(options["dt"])
        self.outcome_type = options.get("outcome_type", "state")

        ss = options.get("seed_sequence", None)
        self.seed_sequence = ss if isinstance(ss, np.random.SeedSequence) else np.random.SeedSequence(ss)
        self.random_state = np.random.default_rng(self.seed_sequence)

        self.electronics = options.get("electronics", None)
        self.hopping = np.zeros(model.nstates(), dtype=np.float64)

        self.electronic_integration = options.get("electronic_integration", "exp").lower()
        self.max_electronic_dt = options.get("max_electronic_dt", 0.1)
        self.starting_electronic_intervals = options.get("starting_electronic_intervals", 4)

        self.weight = float(options.get("weight", 1.0))

        self.restart = options.get("restart", False)
        self.force_quit = False

        self.zeta = 0.0

    ## Update weight held by trajectory and by trace
    def update_weight(self, weight):
        self.weight = weight
        self.tracer.weight = weight

        if self.weight == 0.0:
            self.force_quit = True

    ## Override deepcopy
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        shallow_only = [ "queue" ]
        for k, v in self.__dict__.items():
            setattr(result, k,
                cp.deepcopy(v, memo) if v not in shallow_only else cp.copy(v))
        return result

    def clone(self):
        return cp.deepcopy(self)

    ## Return random number for hopping decisions
    def random(self):
        return self.random_state.uniform()

    ## Is trajectory still inside interaction region?
    def currently_interacting(self):
        """determines whether trajectory is currently inside an interaction region"""
        if self.duration["box_bounds"] is None:
            return False
        return np.all(self.duration["box_bounds"][0] < self.position) and np.all(self.position < self.duration["box_bounds"][1])

    ## Initializes variables related to continue_simulating
    def duration_initialize(self, options):
        """Initializes variables related to continue_simulating"""

        duration = {}
        duration["found_box"] = False

        bounds = options.get('bounds', None)
        if bounds:
            duration["box_bounds"] = ( np.array(bounds[0], dtype=np.float64),
                    np.array(bounds[1], dtype=np.float64) )
        else:
            duration["box_bounds"] = None
        duration["max_steps"] = options.get('max_steps', 100000) # < 0 interpreted as no limit
        duration["max_time"] = options.get('max_time', 1e25) # set to an outrageous number by default

        self.duration = duration

    ## Returns True if a trajectory ought to keep running, False if it should finish
    def continue_simulating(self):
        """Returns True if a trajectory ought to keep running, False if it should finish"""
        if self.force_quit:
            return False
        elif self.duration["max_steps"] >= 0 and self.nsteps >= self.duration["max_steps"]:
            return False
        elif self.time >= self.duration["max_time"] or np.isclose(self.time, self.duration["max_time"], atol=1e-8, rtol=0.0):
            return False
        elif self.duration["found_box"]:
            return self.currently_interacting()
        else:
            if self.currently_interacting():
                self.duration["found_box"] = True
            return True

    ## add results from current time point to tracing function
    def trace(self, force=False):
        if force or (self.nsteps % self.trace_every) == 0:
            self.tracer.collect(self.snapshot())

    ## returns a dictionary with all the loggable data from the trajectory
    def snapshot(self):
        out = {
            "time" : self.time,
            "position"  : np.copy(self.position),
            "momentum"  : self.mass * np.copy(self.velocity),
            "potential" : self.potential_energy(),
            "kinetic"   : self.kinetic_energy(),
            "energy"    : self.total_energy(),
            "density_matrix" : np.copy(self.rho),
            "active"    : self.state,
            "electronics" : self.electronics,
            "hopping"   : self.hopping,
            "zeta"   : self.zeta
            }
        return out

    ## current kinetic energy
    def kinetic_energy(self):
        return 0.5 * np.einsum('m,m,m', self.mass, self.velocity, self.velocity)

    ## current potential energy
    # @param electronics ElectronicStates from current step
    def potential_energy(self, electronics=None):
        if electronics is None:
            electronics = self.electronics
        return electronics.hamiltonian[self.state,self.state]

    ## current kinetic + potential energy
    # @param electronics ElectronicStates from current step
    def total_energy(self, electronics=None):
        potential = self.potential_energy(electronics)
        kinetic = self.kinetic_energy()
        return potential + kinetic

    ## force on active state
    # @param electronics ElectronicStates from current step
    def force(self, electronics=None):
        if electronics is None:
            electronics = self.electronics
        return electronics.force[self.state,:]

    ## Nonadiabatic coupling matrix
    # @param electronics ElectronicStates from current step
    # @param velocity velocity used to compute NAC (defaults to self.velocity)
    def NAC_matrix(self, electronics=None, velocity=None):
        velo = velocity if velocity is not None else self.velocity
        if electronics is None:
            electronics = self.electronics
        return np.einsum("ijx,x->ij", electronics.derivative_coupling, velo)

    ## kinetic energy along given momentum mode
    # @param direction [ndim] numpy array defining direction
    def mode_kinetic_energy(self, direction):
        u = direction / np.linalg.norm(direction)
        momentum = self.velocity * self.mass
        component = np.dot(u, momentum) * u
        return 0.5 * np.einsum('m,m,m', 1.0/self.mass, component, component)

    ## Return direction in which to rescale momentum
    # @param source active state before hop
    # @param target active state after hop
    def direction_of_rescale(self, source, target, electronics=None):
        elec_states = self.electronics if electronics is None else electronics
        out = elec_states.derivative_coupling[source, target, :]
        return out

    ## Rescales velocity in the specified direction and amount
    # @param direction array specifying the direction of the velocity to rescale
    # @param reduction scalar specifying how much kinetic energy should be damped
    def rescale_component(self, direction, reduction):
        # normalize
        direction /= np.sqrt(np.dot(direction, direction))
        M_inv = 1.0 / self.mass
        Md = self.mass * direction
        a = np.einsum('m,m,m', M_inv, direction, direction)
        b = 2.0 * np.dot(self.velocity, direction)
        c = -2.0 * reduction
        roots = np.roots([a, b, c])
        scal = min(roots, key=lambda x: abs(x))
        self.velocity += scal * M_inv * direction

    ## Compute the Hamiltonian used to propagate the electronic wavefunction
    #  returns nonadiabatic coupling H - i W at midpoint between current and previous time steps
    #  @param elec_states ElectronicStates at \f$t\f$
    def hamiltonian_propagator(self, last_electronics, this_electronics, velo=None):
        if velo is None:
            velo = 0.5 * (self.velocity + self.last_velocity)

        H = 0.5 * (this_electronics.hamiltonian + last_electronics.hamiltonian)
        TV = 0.5 * np.einsum("ijx,x->ij", this_electronics.derivative_coupling + last_electronics.derivative_coupling,
                velo)
        return H -1j * TV

    ## Propagates \f$\rho(t)\f$ to \f$\rho(t + dt)\f$
    # @param elec_states ElectronicStates at \f$t\f$
    # @param dt time step
    #
    # The propagation assumes the electronic energies and couplings are static throughout.
    # This will only be true for fairly small time steps
    def propagate_electronics(self, last_electronics, this_electronics, dt):
        if self.electronic_integration == "exp":
            # Use midpoint propagator
            W = self.hamiltonian_propagator(last_electronics, this_electronics)
            diags, coeff = np.linalg.eigh(W)

            # use W as temporary storage
            U = np.linalg.multi_dot([ coeff, np.diag(np.exp(-1j * diags * dt)), coeff.T.conj() ])
            np.dot(U, np.dot(self.rho, U.T.conj(), out=W), out=self.rho)
        elif self.electronic_integration == "linear-rk4":
            last_H = last_electronics.hamiltonian
            this_H = this_electronics.hamiltonian

            last_tau = last_electronics.derivative_coupling
            this_tau = this_electronics.derivative_coupling

            last_v = self.last_velocity
            this_v = self.velocity

            TV00 = np.einsum("ijx,x->ij", last_tau, last_v)
            TV11 = np.einsum("ijx,x->ij", this_tau, this_v)
            TV01 = np.einsum("ijx,x->ij", last_tau, this_v) + np.einsum("ijx,x->ij", this_tau, last_v)

            HH = last_H
            eigs, vecs = np.linalg.eigh(HH)

            H0  = np.linalg.multi_dot([vecs.T, last_H, vecs])
            H1  = np.linalg.multi_dot([vecs.T, this_H, vecs])
            W00 = np.linalg.multi_dot([vecs.T, TV00, vecs])
            W11 = np.linalg.multi_dot([vecs.T, TV11, vecs])
            W01 = np.linalg.multi_dot([vecs.T, TV01, vecs])

            def ydot(rho, t):
                assert t >= 0.0 and t <= dt
                w0 = 1.0 - t/dt
                w1 = t/dt

                ergs = np.exp(1j * eigs * t).reshape([1, -1])
                phases = np.dot(ergs.T, ergs.conj())

                H = H0 * (w0 - 1.0) + H1 * w1
                Hbar = H - 1j * (w0*w0*W00 + w1*w1*W11 + w0*w1*W01)
                HI = Hbar * phases

                out = -1j * ( np.dot(HI, rho) - np.dot(rho, HI) )
                return out

            nsteps = self.starting_electronic_intervals
            while (dt/nsteps > self.max_electronic_dt):
                nsteps *= 2

            rho0 = np.linalg.multi_dot([vecs.T, self.rho, vecs])
            tmprho = rk4(rho0, ydot, 0.0, dt, nsteps)
            ergs = np.exp(1j * eigs * dt).reshape([1, -1])
            phases = np.dot(ergs.T.conj(), ergs)
            self.rho = np.linalg.multi_dot([vecs, tmprho * phases, vecs.T])
        else:
            raise Exception("Unrecognized electronic integration option")

    ## move classical position forward one step
    # @param electronics ElectronicStates from current step
    def advance_position(self, last_electronics, this_electronics):
        acceleration = self.force(this_electronics) / self.mass
        self.last_position = self.position
        self.position += self.velocity * self.dt + 0.5 * acceleration * self.dt * self.dt

    ## move classical velocity forward one step
    # @param electronics ElectronicStates from current step
    def advance_velocity(self, last_electronics, this_electronics):
        last_acceleration = self.force(last_electronics) / self.mass
        this_acceleration = self.force(this_electronics) / self.mass

        self.last_velocity = self.velocity
        self.velocity += 0.5 * (last_acceleration + this_acceleration) * self.dt

    ## Compute probability of hopping, generate random number, and perform hops
    # @param elec_states ElectronicStates from current step
    def surface_hopping(self, last_electronics, this_electronics):
        nstates = self.model.nstates()

        H = self.hamiltonian_propagator(last_electronics, this_electronics)

        probs = 2.0 * np.imag(self.rho[self.state,:] * H[:,self.state]) * self.dt / np.real(self.rho[self.state,self.state])

        # zero out 'self-hop' for good measure (numerical safety)
        probs[self.state] = 0.0

        # clip probabilities to make sure they are between zero and one
        probs = np.maximum(probs, 0.0)
        self.hopping = probs

        hop_targets = self.hopper(probs)
        if hop_targets:
            self.hop_to_it(hop_targets, this_electronics)

        return sum(probs)

    ## given a set of probabilities, determines whether and where to hop
    #
    #  if no hop to be performaned, returns empty list
    #
    # @param probs [nstates] numpy array of individual hopping probabilities
    #  returns [(target_state, weight)]
    def hopper(self, probs):
        self.zeta = self.random()
        acc_prob = np.cumsum(probs)
        hops = np.less(self.zeta, acc_prob)
        if any(hops):
            hop_to = -1
            for i in range(self.model.nstates()):
                if hops[i]:
                    hop_to = i
                    break

            return [{ "target" : hop_to, "weight" : 1.0, "zeta" : self.zeta, "prob" : hops[i] }]
        else:
            return []

    ## Performs the hop from the current active state to the given state, including
    #  rescaling the momentum
    #
    #  @param hop_to final state to hop to
    def hop_to_it(self, hop_targets, electronics=None):
        hop_dict = hop_targets[0]
        hop_to = hop_dict["target"]
        weight = hop_dict["weight"]
        elec_states = electronics if electronics is not None else self.electronics
        new_potential, old_potential = elec_states.hamiltonian[hop_to, hop_to], elec_states.hamiltonian[self.state, self.state]
        delV = new_potential - old_potential
        rescale_vector = self.direction_of_rescale(self.state, hop_to)
        component_kinetic = self.mode_kinetic_energy(rescale_vector)
        if delV <= component_kinetic:
            hop_from = self.state
            self.state = hop_to
            self.rescale_component(rescale_vector, -delV)
            self.tracer.hop(self.time, hop_from, hop_to, hop_dict["zeta"], hop_dict["prob"])

    ## run simulation
    def simulate(self):
        last_electronics = None

        if not self.restart:
            self.electronics = self.model.update(self.position)

        potential_energy = self.potential_energy(self.electronics)

        self.hopping = 0.0

        self.trace()

        # propagation
        while (True):
            # first update nuclear coordinates
            self.advance_position(last_electronics, self.electronics)

            # calculate electronics at new position
            last_electronics, self.electronics = self.electronics, self.electronics.update(self.position)

            # update velocity
            self.advance_velocity(last_electronics, self.electronics)

            # now propagate the electronic wavefunction to the new time
            self.propagate_electronics(last_electronics, self.electronics, self.dt)
            self.hopping = self.surface_hopping(last_electronics, self.electronics)

            self.time += self.dt
            self.nsteps += 1

            # ending condition
            if not self.continue_simulating():
                break

            self.trace()

        self.trace(force=True)

        return self.tracer

    ## Classifies end of simulation:
    #
    #  2*state + [0 for left, 1 for right]
    def outcome(self):
        out = np.zeros([self.model.nstates(), 2], dtype=np.float64)
        lr = 0 if self.position < 0.0 else 1
        if self.outcome_type == "populations":
            out[:,lr] = np.real(self.rho).diag()[:]
        elif self.outcome_type == "state":
            out[self.state,lr] = 1.0
        else:
            raise Exception("Unrecognized outcome recognition type")
        return out
        # first bit is left (0) or right (1), second bit is electronic state

