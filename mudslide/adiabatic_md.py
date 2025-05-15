# -*- coding: utf-8 -*-
"""Propagate Adiabatic MD trajectory"""

from __future__ import division

import copy as cp
import numpy as np

from .constants import fs_to_au, boltzmann

from .tracer import Trace

from typing import List, Dict, Union, Any
from .typing import ElectronicT, ArrayLike, DtypeLike

from .propagator import Propagator_

from .util import remove_center_of_mass_motion, remove_angular_momentum

class VVPropagator(Propagator_):
    """Velocity Verlet propagator"""

    def __init__(self, **options: Any) -> None:
        """Constructor

        :param options: option dictionary
        """
        super().__init__()

    def __call__(self, traj: 'AdiabaticMD', nsteps: int) -> None:
        """Propagate trajectory using Velocity Verlet algorithm

        :param traj: trajectory object to propagate
        :param nsteps: number of steps to propagate
        """
        dt = traj.dt
        # first update nuclear coordinates
        for i in range(nsteps):
            acceleration = traj.force(traj.electronics) / traj.mass
            traj.last_position = traj.position
            traj.position += traj.velocity * dt + 0.5 * acceleration * dt * dt

            # calculate electronics at new position
            traj.last_electronics = traj.electronics
            traj.electronics = traj.model.update(traj.position, electronics=traj.electronics)

            # update velocity
            last_acceleration = acceleration
            acceleration = traj.force(traj.electronics) / traj.mass

            traj.last_velocity = traj.velocity
            traj.velocity += 0.5 * (last_acceleration + acceleration) * dt

            # optionally remove COM motion and total angular momentum
            if traj.remove_com_every > 0 and (traj.nsteps % traj.remove_com_every) == 0:
                d = traj.model.dimensionality
                v = traj.velocity.reshape(d)
                m = traj.mass.reshape(d)[:,0]
                vnew = remove_center_of_mass_motion(v, m)
                traj.velocity = vnew.reshape(traj.velocity.shape)

            if traj.remove_angular_momentum_every > 0 and (traj.nsteps % traj.remove_angular_momentum_every) == 0:
                d = traj.model.dimensionality
                v = traj.velocity.reshape(d)
                m = traj.mass.reshape(d)[:,0]
                x = traj.position.reshape(d)
                vnew = remove_angular_momentum(v, m, x)
                traj.velocity = vnew.reshape(traj.velocity.shape)

            traj.time += dt
            traj.nsteps += 1

class NoseHooverChainPropagator(Propagator_):
    """
    G. J. Martyna, M. E. Tuckerman, D. J. Tobias, and Michael L. Klein,
    "Explicit reversible integrators for extended systems dynamics"
    Molecular Physics, 87, 1117-1157 (1996)
    """

    def __init__(self, temperature: np.float64, timescale: np.float64 = 1e3 * fs_to_au,
                 ndof: int = 3, nchains: int = 3, nys: int = 3, nc: int = 1):
        """Constructor

        :param temperature: thermostat temperature
        :param timescale: thermostat timescale
        :param nchains: number of thermostat chains
        """
        self.temperature = temperature
        self.timescale = timescale
        self.ndof = ndof
        self.nchains = nchains
        self.nys = nys
        self.nc = nc

        assert self.temperature > 0.0
        assert self.timescale > 0.0
        assert self.nchains >= 1
        assert self.nys == 3 or self.nys == 5
        assert self.nc >= 1

        self.nh_position = np.zeros(nchains, dtype=np.float64)
        self.nh_velocity = np.zeros(nchains, dtype=np.float64)
        Q = 2.0 * timescale**2 * boltzmann * temperature
        self.nh_mass = np.ones(nchains, dtype=np.float64) * Q
        self.nh_mass[0] *= ndof

        if nys == 3:
            tmp = 1 / (2 - 2**(1./3))
            self.w = np.array([tmp, 1 - 2*tmp, tmp]) / nc
        else:
            tmp = 1 / (4 - 4**(1./3))
            self.w = np.array([tmp, tmp, 1 - 4*tmp, tmp, tmp]) / nc

        self.G = np.zeros(nchains)


    def nhc_step(self, velocity, mass, dt: float):
        """
        Move forward one step in the extended system variables
        """
        # convenience definitions
        G = self.G
        V = self.nh_velocity
        X = self.nh_position
        M = self.nchains

        scale = 1.0
        K2 = np.sum(velocity * velocity * mass)
        kt = self.temperature * boltzmann
        nkt = self.temperature * boltzmann * self.ndof

        G[0] = (K2 - nkt) / self.nh_mass[0]

        for inc in range(self.nc):
            for iys in range(self.nys):
                wdt = self.w[iys] * dt
                V[M-1] += G[M-1] * wdt / 4.0

                for kk in range(0, M-1):
                    AA = np.exp(-wdt * V[M - 1 - kk] / 8.0)
                    V[M - 2 - kk] = V[M - 2 - kk] * AA * AA \
                                  + wdt * G[M - 2 - kk] * AA / 4.0

                # update the particle velocities
                AA = np.exp(-0.5 * wdt * V[0])
                scale *= AA

                # update the forces
                G[0] = (scale * scale * K2 - nkt) / self.nh_mass[0]

                # update the thermostat positions
                X += 0.5 * V * wdt

                # update the thermostat velocities
                for kk in range(M - 1):
                    AA = np.exp(-0.125 * wdt * V[kk + 1])
                    V[kk] = V[kk] * AA * AA \
                              + 0.25 * wdt * G[kk] * AA
                    G[kk+1] = (self.nh_mass[kk] * V[kk]**2 - kt) / self.nh_mass[kk + 1]
                V[M - 1] += 0.25 * G[M - 1] * wdt

        return scale

    def __call__(self, traj: 'AdiabaticMD', nsteps: int) -> None:
        """Propagate trajectory using Velocity Verlet algorithm with Nose-Hoover thermostat

        :param traj: trajectory object to propagate
        :param nsteps: number of steps to propagate
        """
        dt = traj.dt

        # 0, 1, 2 represents t, t + 0.5*dt, and t + dt, respectively
        for i in range(nsteps):
            vscale = self.nhc_step(traj.velocity, traj.mass, dt)
            acceleration = traj.force(traj.electronics) / traj.mass

            x0 = traj.position
            v1 = traj.velocity * vscale + 0.5 * dt * acceleration

            traj.last_position = x0
            traj.position += v1 * dt

            # calculate electronics at new position
            traj.last_electronics = traj.electronics
            traj.electronics = traj.model.update(traj.position, electronics=traj.electronics)

            last_acceleration = acceleration
            acceleration = traj.force(traj.electronics) / traj.mass

            v2p  = v1 + 0.5 * dt * acceleration

            vscale = self.nhc_step(v2p, traj.mass, dt)
            traj.last_velocity = traj.velocity
            traj.velocity = v2p * vscale

            # optionally remove COM motion and total angular momentum
            if traj.remove_com_every > 0 and (traj.nsteps % traj.remove_com_every) == 0:
                d = traj.model.dimensionality
                v = traj.velocity.reshape(d)
                m = traj.mass.reshape(d)[:,0]
                vnew = remove_center_of_mass_motion(v, m)
                traj.velocity = vnew.reshape(traj.velocity.shape)

            if traj.remove_angular_momentum_every > 0 and (traj.nsteps % traj.remove_angular_momentum_every) == 0:
                d = traj.model.dimensionality
                v = traj.velocity.reshape(d)
                m = traj.mass.reshape(d)[:,0]
                x = traj.position.reshape(d)
                vnew = remove_angular_momentum(v, m, x)
                traj.velocity = vnew.reshape(traj.velocity.shape)

            traj.time += dt
            traj.nsteps += 1

def AdiabaticPropagator(model, prop_options: Any = "vv") -> Propagator_:
    """Factory function for creating propagator objects

    :param prop_type: string or dict propagator type
    :param model: model object (may be necessary to set some defaults)

    :return: propagator object
    """
    if isinstance(prop_options, str):
        prop_options = { "type": prop_options.lower() }

    prop_options["ndof"] = model.ndim()

    if isinstance(prop_options, dict):
        prop_type = prop_options.pop("type", "vv")
        if prop_type in ["vv", "velocity verlet"]:
            return VVPropagator(**prop_options)
        if prop_type in ["nh", "nhc", "nose-hoover"]:
            return NoseHooverChainPropagator(**prop_options)
        else:
            raise ValueError("Unknown propagator type: {}".format(prop_type))

    raise ValueError("Unknown propagator options: {}".format(prop_options))

class AdiabaticMD(object):
    """Class to propagate a single adiabatic trajectory, like ground state MD"""

    def __init__(self,
                 model: Any,
                 x0: ArrayLike,
                 p0: ArrayLike,
                 tracer: Any = None,
                 queue: Any = None,
                 **options: Any):
        """Constructor
        :param model: Model object defining problem
        :param x0: Initial position
        :param p0: Initial momentum
        :param tracer: spawn from TraceManager to collect results
        :param queue: Trajectory queue
        :param options: option dictionary
        """
        self.model = model
        self.tracer = Trace(tracer)
        self.queue: Any = queue
        self.mass = model.mass
        self.position = np.array(x0, dtype=np.float64).reshape(model.ndim())
        self.velocity = np.array(p0, dtype=np.float64).reshape(model.ndim()) / self.mass
        self.last_velocity = np.zeros_like(self.velocity, dtype=np.float64)
        if "last_velocity" in options:
            self.last_velocity[:] = options["last_velocity"]

        # function duration_initialize should get us ready to for future continue_simulating calls
        # that decide whether the simulation has finished
        if "duration" in options:
            self.duration = options["duration"]
        else:
            self.duration_initialize(options)

        # fixed initial parameters
        self.time = np.longdouble(options.get("t0", 0.0))
        self.nsteps = int(options.get("previous_steps", 0))
        self.trace_every = int(options.get("trace_every", 1))

        self.propagator = AdiabaticPropagator(self.model, options.get("propagator", "VV"))

        self.remove_com_every = int(options.get("remove_com_every", 0))
        self.remove_angular_momentum_every = int(options.get("remove_angular_momentum_every", 0))

        # read out of options
        self.dt = np.longdouble(options["dt"])
        self.outcome_type = options.get("outcome_type", "state")

        ss = options.get("seed_sequence", None)
        self.seed_sequence = ss if isinstance(ss, np.random.SeedSequence) else np.random.SeedSequence(ss)
        self.random_state = np.random.default_rng(self.seed_sequence)

        self.electronics = options.get("electronics", None)

        self.weight = np.float64(options.get("weight", 1.0))

        self.restarting = options.get("restarting", False)
        self.force_quit = False

    @classmethod
    def restart(cls, model, log, **options) -> 'AdiabaticMD':
        last_snap = log[-1]
        penultimate_snap = log[-2]

        x = last_snap["position"]
        p = last_snap["momentum"]
        last_velocity = penultimate_snap["momentum"] / model.mass
        t0 = last_snap["time"]
        dt = t0 - penultimate_snap["time"]
        weight = log.weight
        previous_steps = len(log)

        # use inferred data if available, but let kwargs override
        for key, val in [["dt", dt]]:
            if key not in options:
                options[key] = val

        return cls(model,
                   x,
                   p,
                   tracer=log,
                   t0=t0,
                   last_velocity=last_velocity,
                   weight=weight,
                   previous_steps=previous_steps,
                   restarting=True,
                   **options)

    def update_weight(self, weight: np.float64) -> None:
        """Update weight held by trajectory and by trace"""
        self.weight = weight
        self.tracer.weight = weight

        if self.weight == 0.0:
            self.force_quit = True

    def __deepcopy__(self, memo: Any) -> 'AdiabaticMD':
        """Override deepcopy"""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        shallow_only = ["queue"]
        for k, v in self.__dict__.items():
            setattr(result, k, cp.deepcopy(v, memo) if v not in shallow_only else cp.copy(v))
        return result

    def clone(self) -> 'AdiabaticMD':
        """Clone existing trajectory for spawning

        :return: copy of current object
        """
        return cp.deepcopy(self)

    def random(self) -> np.float64:
        """Get random number for hopping decisions

        :return: uniform random number between 0 and 1
        """
        return self.random_state.uniform()

    def currently_interacting(self) -> bool:
        """Determines whether trajectory is currently inside an interaction region

        :return: boolean
        """
        if self.duration["box_bounds"] is None:
            return False
        return np.all(self.duration["box_bounds"][0] < self.position) and np.all(
            self.position < self.duration["box_bounds"][1])

    def duration_initialize(self, options: Dict[str, Any]) -> None:
        """Initializes variables related to continue_simulating

        :param options: dictionary with options
        """

        duration = {}  # type: Dict[str, Any]
        duration['found_box'] = False

        bounds = options.get('bounds', None)
        if bounds:
            duration["box_bounds"] = (np.array(bounds[0], dtype=np.float64), np.array(bounds[1], dtype=np.float64))
        else:
            duration["box_bounds"] = None
        duration["max_steps"] = options.get('max_steps', 100000)  # < 0 interpreted as no limit
        duration["max_time"] = options.get('max_time', 1e25)  # set to an outrageous number by default

        self.duration = duration

    def continue_simulating(self) -> bool:
        """Decide whether trajectory should keep running

        :return: True if trajectory should keep running, False if it should finish
        """
        if self.force_quit:
            return False
        elif self.duration["max_steps"] >= 0 and self.nsteps >= self.duration["max_steps"]:
            return False
        elif self.time >= self.duration["max_time"] or np.isclose(
                self.time, self.duration["max_time"], atol=1e-8, rtol=0.0):
            return False
        elif self.duration["found_box"]:
            return self.currently_interacting()
        else:
            if self.currently_interacting():
                self.duration["found_box"] = True
            return True

    def trace(self, force: bool = False) -> None:
        """Add results from current time point to tracing function
        Only adds snapshot if nsteps%trace_every == 0, unless force=True

        :param force: force snapshot
        """
        if force or (self.nsteps % self.trace_every) == 0:
            self.tracer.collect(self.snapshot())

    def snapshot(self) -> Dict[str, Any]:
        """Collect data from run for logging

        :return: dictionary with all data from current time step
        """
        out = {
            "time": self.time,
            "position": self.position.tolist(),
            "momentum": (self.mass * self.velocity).tolist(),
            "potential": self.potential_energy().item(),
            "kinetic": self.kinetic_energy().item(),
            "energy": self.total_energy().item(),
            "temperature": 2 * self.kinetic_energy() / ( boltzmann * self.model.ndim()),
            "electronics": self.electronics.as_dict()
        }
        return out

    def trouble_shooter(self):
        log = self.snapshot()
        with open("snapout.dat", "a") as file:
            file.write("{}\t{}\t{}\t{}\t{}\n".format(log["time"], log["potential"], log["kinetic"], log["energy"],
                                                     log["active"]))

    def kinetic_energy(self) -> np.float64:
        """Kinetic energy

        :return: kinetic energy
        """
        return 0.5 * np.einsum('m,m,m', self.mass, self.velocity, self.velocity)

    def potential_energy(self, electronics: ElectronicT = None) -> DtypeLike:
        """Potential energy

        :param electronics: ElectronicStates from current step
        :return: potential energy
        """
        if electronics is None:
            electronics = self.electronics
        return electronics.energies[0]

    def total_energy(self, electronics: ElectronicT = None) -> DtypeLike:
        """
        Kinetic energy + Potential energy

        :param electronics: ElectronicStates from current step
        :return: total energy
        """
        potential = self.potential_energy(electronics)
        kinetic = self.kinetic_energy()
        return potential + kinetic

    def force(self, electronics: ElectronicT = None) -> ArrayLike:
        """
        Compute force on active state

        :param electronics: ElectronicStates from current step

        :return: [ndim] force on active electronic state
        """
        if electronics is None:
            electronics = self.electronics
        return electronics.force(0)

    def mode_kinetic_energy(self, direction: ArrayLike) -> np.float64:
        """
        Kinetic energy along given momentum mode

        :param direction: [ndim] numpy array defining direction

        :return: kinetic energy along specified direction
        """
        u = direction / np.linalg.norm(direction)
        momentum = self.velocity * self.mass
        component = np.dot(u, momentum) * u
        return 0.5 * np.einsum('m,m,m', 1.0 / self.mass, component, component)

    def simulate(self) -> 'Trace':
        """
        Simulate

        :return: Trace of trajectory
        """

        if not self.continue_simulating():
            return self.tracer

        self.last_electronics = None

        if self.electronics is None:
            self.electronics = self.model.update(self.position)

        if not self.restarting:
            self.trace()

        # propagation
        while (True):
            self.propagator(self, 1)

            # ending condition
            if not self.continue_simulating():
                break

            self.trace()

        self.trace(force=True)

        return self.tracer
