# -*- coding: utf-8 -*-
"""Propagate Adiabatic MD trajectory"""

import copy as cp
from typing import Dict, Union, Any

import numpy as np

from .tracer import DefaultTrace
from .typing import ElectronicT, ArrayLike, DtypeLike

class AdiabaticMD:
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
        self.tracer = tracer if tracer is not None else DefaultTrace()
        self.queue: Any = queue
        self.mass = model.mass
        self.position = np.array(x0, dtype=np.float64).reshape(model.ndim())
        self.velocity = np.array(p0, dtype=np.float64).reshape(model.ndim()) / self.mass
        self.last_position = np.zeros_like(self.position, dtype=np.float64)
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

        self.weight = float(options.get("weight", 1.0))

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

    def update_weight(self, weight: float) -> None:
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
        with open("snapout.dat", "a", encoding='utf-8') as file:
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

    def advance_position(self, last_electronics: Union[ElectronicT, None], this_electronics: ElectronicT) -> None:
        """
        Move classical position forward one step

        :param last_electronics: ElectronicStates from previous step
        :param this_electronics: ElectronicStates from current step
        """
        acceleration = self.force(this_electronics) / self.mass
        self.last_position = self.position
        self.position += self.velocity * self.dt + 0.5 * acceleration * self.dt * self.dt

    def advance_velocity(self, last_electronics: ElectronicT, this_electronics: ElectronicT) -> None:
        """
        Move classical velocity forward one step

        :param electronics: ElectronicStates from current step
        """

        last_acceleration = self.force(last_electronics) / self.mass
        this_acceleration = self.force(this_electronics) / self.mass

        self.last_velocity = self.velocity
        self.velocity += 0.5 * (last_acceleration + this_acceleration) * self.dt

    def simulate(self) -> 'Trace':
        """
        Simulate

        :return: Trace of trajectory
        """

        if not self.continue_simulating():
            return self.tracer

        last_electronics = None

        if self.electronics is None:
            self.electronics = self.model.update(self.position)

        if not self.restarting:
            self.trace()

        # propagation
        while True:
            # first update nuclear coordinates
            self.advance_position(last_electronics, self.electronics)

            # calculate electronics at new position
            last_electronics, self.electronics = self.electronics, self.model.update(self.position, electronics=self.electronics)

            # update velocity
            self.advance_velocity(last_electronics, self.electronics)

            self.time += self.dt
            self.nsteps += 1

            # ending condition
            if not self.continue_simulating():
                break

            self.trace()

        self.trace(force=True)

        return self.tracer
