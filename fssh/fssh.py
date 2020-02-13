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

## Class to propagate a single SH Trajectory
class TrajectorySH(object):
    ## Constructor
    # @param model Model object defining problem
    # @param tracer spawn from TraceManager to collect results
    # @param options option dictionary
    def __init__(self, model, x0, p0, initial, tracer=None, **options):
        self.model = model
        self.tracer = tracer if tracer is not None else Trace()
        self.mass = model.mass
        self.position = np.array(x0, dtype=np.float64).reshape(model.ndim())
        self.velocity = np.array(p0, dtype=np.float64).reshape(model.ndim()) / self.mass
        self.last_velocity = np.zeros_like(self.velocity, dtype=np.float64)
        if initial == "ground":
            self.rho = np.zeros([model.nstates(),model.nstates()], dtype=np.complex128)
            self.rho[0,0] = 1.0
            self.state = 0
        else:
            raise Exception("Unrecognized initial state option")

        # function duration_initialize should get us ready to for future continue_simulating calls
        # that decide whether the simulation has finished
        self.duration_initialize(options)

        # fixed initial parameters
        self.time = options.get("t0", 0.0)
        self.nsteps = options.get("previous_steps", 0)
        self.trace_every = options.get("trace_every", 1)

        # read out of options
        self.dt = options["dt"]
        self.outcome_type = options.get("outcome_type", "state")

        self.random_state = np.random.RandomState(options.get("seed", None))

        self.electronics = None
        self.hopping = np.zeros(model.nstates(), dtype=np.float64)

    ## Return random number for hopping decisions
    def random(self):
        return self.random_state.uniform()

    ## Is trajectory still inside interaction region?
    def currently_interacting(self):
        """determines whether trajectory is currently inside an interaction region"""
        if self.box_bounds is None:
            return False
        return np.all(self.box_bounds[0] < self.position) and np.all(self.position < self.box_bounds[1])

    ## Initializes variables related to continue_simulating
    def duration_initialize(self, options):
        """Initializes variables related to continue_simulating"""
        self.found_box = False

        bounds = options.get('bounds', None)
        if bounds:
            self.box_bounds = ( np.array(bounds[0], dtype=np.float64),
                    np.array(bounds[1], dtype=np.float64) )
        else:
            self.box_bounds = None
        self.max_steps = options.get('max_steps', 10000)
        self.max_time = options.get('max_time', 1e25) # default is to hopefully never hit this

    ## Returns True if a trajectory ought to keep running, False if it should finish
    def continue_simulating(self):
        """Returns True if a trajectory ought to keep running, False if it should finish"""
        if self.nsteps > self.max_steps or self.time > self.max_time:
            return False
        elif self.found_box:
            return self.currently_interacting()
        else:
            if self.currently_interacting():
                self.found_box = True
            return True

    ## add results from current time point to tracing function
    def trace(self):
        if (self.nsteps % self.trace_every) == 0:
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
            "hopping"   : self.hopping
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

    ## kinetic energy along given mode
    # @param direction [ndim] numpy array defining direction
    def mode_kinetic_energy(self, direction):
        component = np.dot(direction, self.velocity) / np.dot(direction, direction) * direction
        return 0.5 * np.einsum('m,m,m', self.mass, component, component)

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
        M_inv = 1.0 / self.mass
        Md = self.mass * direction
        a = np.einsum('m,m,m', M_inv, direction, direction)
        b = 2.0 * np.dot(self.velocity, direction)
        c = -2.0 * reduction
        roots = np.roots([a, b, c])
        scal = min(roots, key=lambda x: abs(x))
        self.velocity += scal * M_inv * direction

    ## Compute the Hamiltonian used to propagate the electronic wavefunction
    #  returns nonadiabatic coupling H - i W
    #  @param elec_states ElectronicStates at \f$t\f$
    def hamiltonian_propagator(self, elec_states):
        velo = 0.5*(self.last_velocity + self.velocity)

        out = elec_states.hamiltonian - 1j * self.NAC_matrix(elec_states, velo)
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
        W = self.NAC_matrix(elec_states, velo)[self.state, :]

        probs = 2.0 * np.real(self.rho[self.state,:]) * W[:] * self.dt / np.real(self.rho[self.state,self.state])

        # zero out 'self-hop' for good measure (numerical safety)
        probs[self.state] = 0.0

        # clip probabilities to make sure they are between zero and one
        probs = probs.clip(0.0, 1.0)
        self.hopping = probs

        do_hop, hop_to = self.hopper(probs)
        if (do_hop): # do switch
            new_potential, old_potential = elec_states.hamiltonian[hop_to, hop_to], elec_states.hamiltonian[self.state, self.state]
            delV = new_potential - old_potential
            derivative_coupling = elec_states.derivative_coupling[hop_to, self.state, :]
            component_kinetic = self.mode_kinetic_energy(derivative_coupling)
            if delV <= component_kinetic:
                hop_from = self.state
                self.state = hop_to
                u = self.rescale_direction(derivative_coupling, self.state, hop_to)
                self.rescale_component(u, -delV)
                self.tracer.hop(self.time, hop_from, hop_to)

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

    ## run simulation
    def simulate(self):
        last_electronics = None
        self.electronics = self.model.update(self.position)

        # start by taking half step in velocity
        initial_acc = self.force(self.electronics) / self.mass
        veloc = self.velocity
        dv = 0.5 * initial_acc * self.dt
        self.last_velocity, self.velocity = veloc - dv, veloc + dv

        # propagate wavefunction a half-step forward to match velocity
        self.propagate_electronics(self.electronics, 0.5*self.dt)
        potential_energy = self.potential_energy(self.electronics)

        self.hopping = 0.0

        self.trace()

        # propagation
        while (True):
            # first update nuclear coordinates
            self.advance_position(self.electronics)

            # calculate electronics at new position
            last_electronics, self.electronics = self.electronics, self.electronics.update(self.position)

            # update velocity
            self.advance_velocity(self.electronics)

            # now propagate the electronic wavefunction to the new time
            self.propagate_electronics(self.electronics, self.dt)
            self.hopping = self.surface_hopping(self.electronics)

            self.time += self.dt
            self.nsteps += 1

            # ending condition
            if not self.continue_simulating():
                break

            self.trace()

        self.trace()

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

## Collect results from a single trajectory
class Trace(object):
    def __init__(self):
        self.data = []
        self.hops = []

    ## collect and optionally process data
    def collect(self, trajectory_snapshot):
        self.data.append(trajectory_snapshot)

    def hop(self, time, hop_from, hop_to):
        self.hops.append({
            "time" : time,
            "from" : hop_from,
            "to"   : hop_to
            })

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

    ## returns loggable data
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
            "prob_cum"  : self.prob_cum
            }
        return out

    ## given a set of probabilities, determines whether and where to hop
    # @param probs [nstates] numpy array of individual hopping probabilities
    # TODO this is bad for multiple states: won't be reproducible
    #  returns (do_hop, target_state)
    def hopper(self, probs):
        accumulated = self.prob_cum
        for i, p in enumerate(probs):
            accumulated = accumulated * (1.0 - p) + (1.0 - accumulated) * p
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

