# -*- coding: utf-8 -*-
"""Propagate Adiabatic MD trajectory"""

from __future__ import division

import copy as cp
import numpy as np

from .constants import fs_to_au

from .tracer import Trace

from typing import List, Dict, Union, Any
from .typing import ElectronicT, ArrayLike, DtypeLike

from .propagator import Propagator_

class VVPropagator(Propagator_):
    """Velocity Verlet propagator"""

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

            traj.time += dt
            traj.nsteps += 1

class NoseHooverChainPropagator(Propagator_):
    """
    G. J. Martyna, M. E. Tuckerman, D. J. Tobias, and Michael L. Klein,
    "Explicit reversible integrators for extended systems dynamics"
    Molecular Physics, 87, 1117-1157 (1996)
    """

    def __init__(self, temperature: np.float64, timescale: np.float64 = 1e5 * fs_to_au,
                 nchains:int = 2, nys: int = 3, nc: int = 1):
        """Constructor

        :param temperature: thermostat temperature
        :param timescale: thermostat timescale
        :param nchains: number of thermostat chains
        """
        self.temperature = temperature
        self.timescale = timescale
        self.nchains = nchains
        self.nys = nys
        self.nc = nc

        assert self.temperature > 0.0
        assert self.timescale > 0.0
        assert self.nchains >= 1
        assert self.nys == 3 or self.nys == 5
        assert self.nc >= 1

        self.nh_position = np.zeros(nchains, dtype=np.float64)
        self.nh_momentum = np.zeros(nchains, dtype=np.float64)
        self.nh_mass = np.ones(nchains, dtype=np.float64) / self.timescale

        if nys == 3:
            tmp = 1 / (2 - 2**(1./3))
            self.wdti = np.array([tmp, 1 - 2*tmp, tmp]) * dt / nc
        else:
            tmp = 1 / (4 - 4**(1./3))
            self.wdti = np.array([tmp, tmp, 1 - 4*tmp, tmp, tmp]) * dt / nc

        self.Vlogs = np.zeros(M)
        self.Xlogs = np.zeros(M)
        self.Glogs = np.zeros(M)


    def xstep(self, velocity, mass):
        """
        Move forward one step in the extended system variables
        """
        scale = 1.0
        K2 = np.sum(velocity * velocity * mass)
        Glogs[0] = (K2 - self.temperature) / self.nhc_mass[0]

        M = self.nchains

        Glogs = self.Glogs
        Vlogs = self.Vlogs
        Xlogs = self.Xlogs

        for inc in range(nc):
            for iys in range(nys):
                wdt = self.wdti[iys]
                # update the thermostat velocities
                Vlogs[self.nchains - 1] += 0.25 * Glogs[M - 1] * wdt

                for kk in range(M - 1):
                    AA = np.exp(-0.125 * wdt * Vlogs[M - 1 - kk])
                    Vlogs[M - 2 - kk] = Vlogs[M - 2 - kk] * AA * AA \
                                  + 0.25 * wdt * Glogs[M - 2 - kk] * AA

                # update the particle velocities
                AA = np.exp(-0.5 * wdt * Vlogs[0])
                scale *= AA
                # update the forces
                Glogs[0] = (scale * scale * K2 - T) / self.nh_mass[0]
                # update the thermostat positions
                Xlogs += 0.5 * Vlogs * wdt
                # update the thermostat velocities
                for kk in range(M - 1):
                    AA = np.exp(-0.125 * wdt * Vlogs[kk + 1])
                    Vlogs[kk] = Vlogs[kk] * AA * AA \
                              + 0.25 * wdt * Glogs[kk] * AA
                    Glogs[kk+1] = (self.nh_mass[kk] * Vlogs[kk]**2 - T) / self.nh_mass[kk + 1]
                Vlogs[M - 1] += 0.25 * Glogs[M - 1] * wdt

        return scale

    def __call__(self, traj: 'AdiabaticMD', nsteps: int) -> None:
        """Propagate trajectory using Velocity Verlet algorithm with Nose-Hoover thermostat

        :param traj: trajectory object to propagate
        :param nsteps: number of steps to propagate
        """
        dt = traj.dt

        # 0, 1, 2 represents t, t + 0.5*dt, and t + dt, respectively
        for i in range(nsteps):
            vscale = self.xstep(traj.velocity, traj.mass)
            acceleration = traj.force(traj.electronics) / traj.mass

            v1 = traj.velocity * vscale + 0.5 * traj.dt * acceleration

            traj.last_position = traj.position
            traj.position += v1 * traj.dt
            x2   = x0 + v1 * dt

            # calculate electronics at new position
            traj.last_electronics = traj.electronics
            traj.electronics = traj.model.update(traj.position, electronics=traj.electronics)

            last_acceleration = acceleration
            acceleration = traj.force(traj.electronics) / traj.mass

            v2p  = v1 + 0.5 * dt * acceleration

            vscale = self.xstep(v2p, traj.mass)
            traj.last_velocity = traj.velocity
            traj.velocity = v2p * vscale

            traj.time += dt
            traj.nsteps += 1

def AdiabaticPropagator(prop_options: Any = "VV") -> Propagator_:
    """Factory function for creating propagator objects

    :param prop_type: string or dict propagator type
    """
    if isinstance(prop_options, str):
        if prop_options == "VV":
            return VVPropagator()
        else:
            raise ValueError("Unknown propagator type: {}".format(prop_options))
    elif isinstance(prop_options, dict):
        prop_type = prop_options["type"]
        if prop_type == "VV":
            return VVPropagator(**prop_options)
        else:
            raise ValueError("Unknown propagator type: {}".format(prop_type))


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

        self.propagator = AdiabaticPropagator(options.get("propagator", "VV"))

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
