# -*- coding: utf-8 -*-
"""Propagate Adiabatic MD trajectory.

This module provides functionality for propagating adiabatic molecular dynamics trajectories,
similar to ground state molecular dynamics simulations.
"""

from typing import Dict, Any
import copy as cp

import numpy as np
from numpy.typing import ArrayLike

from .util import check_options
from .constants import boltzmann
from .tracer import Trace
from .adiabatic_propagator import AdiabaticPropagator

class AdiabaticMD:
    """Class to propagate a single adiabatic trajectory, like ground state MD.

    This class handles the propagation of molecular dynamics trajectories in the
    adiabatic regime, similar to ground state molecular dynamics.
    """
    recognized_options = [
        "dt", "t0", "trace_every", "remove_com_every", "remove_angular_momentum_every",
        "max_steps", "max_time", "bounds", "propagator", "seed_sequence", "electronics",
        "outcome_type", "weight"
    ]

    def __init__(self,
                 model: Any,
                 x0: ArrayLike,
                 v0: ArrayLike,
                 tracer: Any = None,
                 queue: Any = None,
                 strict_option_check: bool = True,
                 **options: Any):
        """Initialize the AdiabaticMD class.

        Parameters
        ----------
        model : Any
            Model object defining problem.
        x0 : ArrayLike
            Initial position.
        v0 : ArrayLike
            Initial velocity.
        tracer : Any, optional
            Spawn from TraceManager to collect results.
        queue : Any, optional
            Trajectory queue.
        strict_option_check : bool, optional
            Whether to strictly check options.
        **options : Any
            Additional options for the simulation. Recognized options are:

            dt : float, optional
                Time step for nuclear propagation (in atomic units). Required.
            t0 : float, optional
                Initial time. Default is 0.0.
            trace_every : int, optional
                Interval (in steps) at which to record trajectory data. Default is 1.
            remove_com_every : int, optional
                Interval for removing center-of-mass motion. Default is 0 (never).
            remove_angular_momentum_every : int, optional
                Interval for removing angular momentum. Default is 0 (never).
            max_steps : int, optional
                Maximum number of steps. Default is 100000.
            max_time : float, optional
                Maximum simulation time. Default is 1e25.
            bounds : tuple or list, optional
                Tuple or list of (lower, upper) bounds for the simulation box. Default is None.
            propagator : str or dict, optional
                The propagator to use for nuclear motion. Can be a string (e.g., 'VV') or a
                dictionary with more options. Default is 'VV'.
            seed_sequence : int or numpy.random.SeedSequence, optional
                Seed or SeedSequence for random number generation. Default is None.
            electronics : object, optional
                Initial electronic state object. Default is None.
            outcome_type : str, optional
                Type of outcome to record (e.g., 'state'). Default is 'state'.
            weight : float, optional
                Statistical weight of the trajectory. Default is 1.0.
            restarting : bool, optional
                Whether this is a restarted trajectory. Default is False.
        """
        check_options(options, self.recognized_options, strict=strict_option_check)

        self.model = model
        self.tracer = Trace(tracer)
        self.queue: Any = queue
        self.mass = model.mass
        self.position = np.array(x0, dtype=np.float64).reshape(model.ndof)
        self.velocity = np.array(v0, dtype=np.float64).reshape(model.ndof)
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
        self.time = np.longdouble(options.get("t0", 0.0))
        self.nsteps = int(options.get("previous_steps", 0))
        self.max_steps = int(options.get("max_steps", 100000))
        self.max_time = float(options.get("max_time", 1e25))
        self.trace_every = int(options.get("trace_every", 1))

        self.propagator = AdiabaticPropagator(self.model, options.get("propagator", "VV"))

        self.remove_com_every = int(options.get("remove_com_every", 0))
        self.remove_angular_momentum_every = int(options.get("remove_angular_momentum_every", 0))

        # read out of options
        self.dt = np.longdouble(options["dt"])
        self.outcome_type = options.get("outcome_type", "state")

        ss = options.get("seed_sequence", None)
        self.seed_sequence = ss if isinstance(ss, np.random.SeedSequence) \
                else np.random.SeedSequence(ss)
        self.random_state = np.random.default_rng(self.seed_sequence)

        self.electronics = options.get("electronics", None)
        self.last_electronics = None

        self.weight = np.float64(options.get("weight", 1.0))

        self.restarting = options.get("restarting", False)
        self.force_quit = False

    @classmethod
    def restart(cls, model, log, **options) -> 'AdiabaticMD':
        """Restart trajectory from log.

        Parameters
        ----------
        model : Any
            Model object defining problem.
        log : Trace
            Trace object with previous trajectory.
        **options : Any
            Additional options for the simulation.

        Returns
        -------
        AdiabaticMD
            New AdiabaticMD object initialized from the log.
        """
        last_snap = log[-1]
        penultimate_snap = log[-2]

        x = last_snap["position"]
        v = np.array(last_snap["velocity"])
        last_velocity = np.array(penultimate_snap["velocity"])
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
                   v,
                   tracer=log,
                   t0=t0,
                   last_velocity=last_velocity,
                   weight=weight,
                   previous_steps=previous_steps,
                   restarting=True,
                   **options)

    def update_weight(self, weight: np.float64) -> None:
        """Update weight held by trajectory and by trace.

        Parameters
        ----------
        weight : np.float64
            New weight value to set.
        """
        self.weight = weight
        self.tracer.weight = weight

        if self.weight == 0.0:
            self.force_quit = True

    def __deepcopy__(self, memo: Any) -> 'AdiabaticMD':
        """Override deepcopy.

        Parameters
        ----------
        memo : Any
            Memo dictionary for deepcopy.

        Returns
        -------
        AdiabaticMD
            Deep copy of the current object.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        shallow_only = ["queue"]
        for k, v in self.__dict__.items():
            setattr(result, k, cp.deepcopy(v, memo) if v not in shallow_only else cp.copy(v))
        return result

    def clone(self) -> 'AdiabaticMD':
        """Clone existing trajectory for spawning.

        Returns
        -------
        AdiabaticMD
            Copy of current object.
        """
        return cp.deepcopy(self)

    def random(self) -> np.float64:
        """Get random number for hopping decisions.

        Returns
        -------
        np.float64
            Uniform random number between 0 and 1.
        """
        return self.random_state.uniform()

    def currently_interacting(self) -> bool:
        """Determine whether trajectory is currently inside an interaction region.

        Returns
        -------
        bool
            True if trajectory is inside interaction region, False otherwise.
        """
        if self.duration["box_bounds"] is None:
            return False
        return np.all(self.duration["box_bounds"][0] < self.position) and np.all(
            self.position < self.duration["box_bounds"][1])

    def duration_initialize(self, options: Dict[str, Any]) -> None:
        """Initialize variables related to continue_simulating.

        Parameters
        ----------
        options : Dict[str, Any]
            Dictionary with options.
        """

        duration = {}  # type: Dict[str, Any]
        duration['found_box'] = False

        bounds = options.get('bounds', None)
        if bounds:
            b0 = np.array(bounds[0], dtype=np.float64)
            b1 = np.array(bounds[1], dtype=np.float64)
            duration["box_bounds"] = (b0, b1)
        else:
            duration["box_bounds"] = None

        self.duration = duration

    def continue_simulating(self) -> bool:
        """Decide whether trajectory should keep running.

        Returns
        -------
        bool
            True if trajectory should keep running, False if it should finish.
        """
        if self.force_quit: # pylint: disable=no-else-return
            return False
        elif self.max_steps >= 0 and self.nsteps >= self.max_steps:
            return False
        elif self.time >= self.max_time or np.isclose(
                self.time, self.max_time, atol=1e-8, rtol=0.0):
            return False
        elif self.duration["found_box"]:
            return self.currently_interacting()
        else:
            if self.currently_interacting():
                self.duration["found_box"] = True
            return True

    def trace(self, force: bool = False) -> None:
        """Add results from current time point to tracing function.

        Only adds snapshot if nsteps%trace_every == 0, unless force=True.

        Parameters
        ----------
        force : bool, optional
            Force snapshot regardless of trace_every interval.
        """
        if force or (self.nsteps % self.trace_every) == 0:
            self.tracer.collect(self.snapshot())

    def snapshot(self) -> Dict[str, Any]:
        """Collect data from run for logging.

        Returns
        -------
        Dict[str, Any]
            Dictionary with all data from current time step.
        """
        out = {
            "time": self.time,
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "potential": self.potential_energy().item(),
            "kinetic": self.kinetic_energy().item(),
            "energy": self.total_energy().item(),
            "temperature": 2 * self.kinetic_energy() / ( boltzmann * self.model.ndof),
            "electronics": self.electronics.as_dict()
        }
        return out

    def kinetic_energy(self) -> np.float64:
        """Calculate kinetic energy.

        Returns
        -------
        np.float64
            Kinetic energy.
        """
        return 0.5 * np.einsum('m,m,m', self.mass, self.velocity, self.velocity)

    def potential_energy(self, electronics: 'ElectronicModel_' = None) -> np.floating:
        """Calculate potential energy.

        Parameters
        ----------
        electronics : ElectronicModel, optional
            Electronic states from current step.

        Returns
        -------
        np.floating
            Potential energy.
        """
        if electronics is None:
            electronics = self.electronics
        return electronics.energies[0]

    def total_energy(self, electronics: 'ElectronicModel_' = None) -> np.floating:
        """Calculate total energy (kinetic + potential).

        Parameters
        ----------
        electronics : ElectronicModel, optional
            Electronic states from current step.

        Returns
        -------
        np.floating
            Total energy.
        """
        potential = self.potential_energy(electronics)
        kinetic = self.kinetic_energy()
        return potential + kinetic

    def force(self, electronics: 'ElectronicModel_' = None) -> ArrayLike:
        """Compute force on active state.

        Parameters
        ----------
        electronics : 'ElectronicModel', optional
            ElectronicStates from current step.

        Returns
        -------
        ArrayLike
            Force on active electronic state.
        """
        if electronics is None:
            electronics = self.electronics
        return electronics.force(0)

    def mode_kinetic_energy(self, direction: ArrayLike) -> np.float64:
        """Calculate kinetic energy along given momentum mode.

        Parameters
        ----------
        direction : ArrayLike
            Array defining direction.

        Returns
        -------
        np.float64
            Kinetic energy along specified direction.
        """
        u = direction / np.linalg.norm(direction)
        momentum = self.velocity * self.mass
        component = np.dot(u, momentum) * u
        return 0.5 * np.einsum('m,m,m', 1.0 / self.mass, component, component)

    def simulate(self) -> 'Trace':
        """Run the simulation.

        Returns
        -------
        Trace
            Trace of trajectory.
        """

        if not self.continue_simulating():
            return self.tracer

        if self.electronics is None:
            self.electronics = self.model.update(self.position)

        if not self.restarting:
            self.trace()

        # propagation
        while True:
            self.propagator(self, 1) # pylint: disable=not-callable

            # ending condition
            if not self.continue_simulating():
                break

            self.trace()

        self.trace(force=True)

        return self.tracer
