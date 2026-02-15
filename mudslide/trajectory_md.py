# -*- coding: utf-8 -*-
"""Base class for MD trajectory propagation.

This module provides the abstract base class TrajectoryMD, which contains
shared infrastructure for all molecular dynamics trajectory types, including
adiabatic, surface hopping, and Ehrenfest dynamics.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Dict, Any, TYPE_CHECKING
import copy as cp

import numpy as np

from .util import check_options
from .constants import boltzmann, fs_to_au
from .tracer import Trace, Trace_
from .propagator import Propagator_

if TYPE_CHECKING:
    from .models.electronics import ElectronicModel_


TIME_COMPARISON_ATOL: float = 1e-12


class TrajectoryMD(ABC):  # pylint: disable=too-many-instance-attributes
    """Abstract base class for molecular dynamics trajectories.

    This class provides shared infrastructure for all trajectory types,
    including common initialization, simulation loop, and analysis methods.
    Subclasses must implement force and potential energy calculations, as
    well as a factory method for creating the appropriate propagator.
    """

    recognized_options: list[str] = [
        "dt", "t0", "trace_every", "remove_com_every",
        "remove_angular_momentum_every", "max_steps", "max_time", "bounds",
        "propagator", "seed_sequence", "electronics", "outcome_type",
        "weight", "last_velocity", "previous_steps", "restarting", "duration",
        "report_every", "report_file",
    ]

    def __init__(self,
                 model: Any,
                 x0: np.ndarray,
                 v0: np.ndarray,
                 tracer: Any = None,
                 queue: Any = None,
                 strict_option_check: bool = True,
                 **options: Any):
        """Initialize common trajectory state.

        Parameters
        ----------
        model : Any
            Model object defining problem.
        x0 : np.ndarray
            Initial position.
        v0 : np.ndarray
            Initial velocity.
        tracer : Any, optional
            Spawn from TraceManager to collect results.
        queue : Any, optional
            Trajectory queue.
        strict_option_check : bool, optional
            Whether to strictly check options.
        **options : Any
            Additional options for the simulation. Recognized options are:

            dt : float
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
                Maximum number of steps. Default depends on subclass.
            max_time : float, optional
                Maximum simulation time. Default is 1e25.
            bounds : tuple or list, optional
                Tuple or list of (lower, upper) bounds for the simulation box.
                Default is None.
            propagator : str or dict, optional
                The propagator to use for nuclear motion. Default depends on subclass.
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
        check_options(options,
                      self.recognized_options,
                      strict=strict_option_check)

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

        # function duration_initialize should get us ready for future
        # continue_simulating calls that decide whether the simulation
        # has finished
        if "duration" in options:
            self.duration = options["duration"]
        else:
            self.duration_initialize(options)

        # fixed initial parameters
        self.time = float(options.get("t0", 0.0))
        self.nsteps = int(options.get("previous_steps", 0))
        self.max_steps = int(
            options.get("max_steps", 100000))
        self.max_time = float(options.get("max_time", 1e25))
        self.trace_every = int(options.get("trace_every", 1))
        self.remove_com_every = int(options.get("remove_com_every", 0))
        self.remove_angular_momentum_every = int(
            options.get("remove_angular_momentum_every", 0))
        if "dt" not in options:
            raise ValueError("dt option is required")
        self.dt = float(options["dt"])

        self.propagator: Propagator_ = self.make_propagator(
            model, options)  # type: ignore[assignment]

        self.outcome_type = options.get("outcome_type", "state")

        ss = options.get("seed_sequence", None)
        self.seed_sequence = ss if isinstance(ss, np.random.SeedSequence) \
                else np.random.SeedSequence(ss)
        self.random_state = np.random.default_rng(self.seed_sequence)

        self.electronics: ElectronicModel_ | None = options.get(
            "electronics", None)
        self.last_electronics: ElectronicModel_ | None = options.get(
            "last_electronics", None)

        self.weight = float(options.get("weight", 1.0))

        self.restarting = options.get("restarting", False)
        self.force_quit = False

        # Progress reporting
        self.report_file: str | None = options.get("report_file",
                                                    "mudslide.report")
        report_every_opt = options.get("report_every", None)
        if report_every_opt is not None:
            self.report_every = int(report_every_opt)
        else:
            total_steps = self.max_steps
            if total_steps <= 0 or total_steps > 1e15:
                total_steps = int(self.max_time / self.dt) if self.max_time < 1e15 else 0
            self.report_every = max(1, total_steps // 100) if total_steps > 0 else 0
        self._temp_buffer: deque[float] = deque(maxlen=50)
        self._last_report_time: float = 0.0

    @abstractmethod
    def make_propagator(self, model: Any,
                        options: Dict[str, Any]) -> Propagator_:
        """Create the propagator for this trajectory type.

        Parameters
        ----------
        model : Any
            Model object defining problem.
        options : Dict[str, Any]
            Options dictionary.

        Returns
        -------
        Propagator_
            Propagator instance for this trajectory.
        """

    def update_weight(self, weight: float) -> None:
        """Update weight held by trajectory and by trace.

        Parameters
        ----------
        weight : float
            New weight value to set.
        """
        self.weight = weight
        self.tracer.weight = weight

        if self.weight == 0.0:
            self.force_quit = True

    def __deepcopy__(self, memo: Any) -> 'TrajectoryMD':
        """Override deepcopy.

        Parameters
        ----------
        memo : Any
            Memo dictionary for deepcopy.

        Returns
        -------
        TrajectoryMD
            Deep copy of the current object.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        shallow_only = ["queue"]
        for k, v in self.__dict__.items():
            setattr(
                result, k,
                cp.deepcopy(v, memo) if k not in shallow_only else cp.copy(v))
        return result

    def clone(self) -> 'TrajectoryMD':
        """Clone existing trajectory for spawning.

        Returns
        -------
        TrajectoryMD
            Copy of current object.
        """
        return cp.deepcopy(self)

    def random(self) -> float:
        """Get random number.

        Returns
        -------
        float
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
        return bool(
            np.all(self.duration["box_bounds"][0] < self.position) and np.all(
                self.position < self.duration["box_bounds"][1]))

    def duration_initialize(self, options: Dict[str, Any]) -> None:
        """Initialize variables related to continue_simulating.

        Parameters
        ----------
        options : Dict[str, Any]
            Dictionary with options.
        """
        duration: Dict[str, Any] = {}
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
        if self.force_quit:
            return False
        if self.max_steps >= 0 and self.nsteps >= self.max_steps:
            return False
        if self.time >= self.max_time or np.isclose(
                self.time, self.max_time, atol=TIME_COMPARISON_ATOL, rtol=0.0):
            return False
        if self.duration["found_box"]:
            return self.currently_interacting()
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

        Subclasses may override to add additional fields.

        Returns
        -------
        Dict[str, Any]
            Dictionary with all data from current time step.
        """
        assert self.electronics is not None
        out = {
            "time":
                float(self.time),
            "position":
                self.position.tolist(),
            "velocity":
                self.velocity.tolist(),
            "potential":
                float(self.potential_energy()),
            "kinetic":
                float(self.kinetic_energy()),
            "energy":
                float(self.total_energy()),
            "temperature":
                float(2 * self.kinetic_energy() /
                      (boltzmann * self.model.ndof)),
            "electronics":
                self.electronics.as_dict()
        }
        return out

    def kinetic_energy(self) -> np.float64:
        """Calculate kinetic energy.

        Returns
        -------
        np.float64
            Kinetic energy.
        """
        return 0.5 * np.einsum('m,m,m', self.mass, self.velocity,
                               self.velocity)

    @abstractmethod
    def potential_energy(self,
                         electronics: ElectronicModel_ | None = None
                        ) -> float:
        """Calculate potential energy.

        Parameters
        ----------
        electronics : ElectronicModel_, optional
            Electronic states from current step.

        Returns
        -------
        float
            Potential energy.
        """

    def total_energy(self,
                     electronics: ElectronicModel_ | None = None) -> float:
        """Calculate total energy (kinetic + potential).

        Parameters
        ----------
        electronics : ElectronicModel_, optional
            Electronic states from current step.

        Returns
        -------
        float
            Total energy.
        """
        potential = self.potential_energy(electronics)
        kinetic = self.kinetic_energy()
        return potential + kinetic

    @abstractmethod
    def force(self,
              electronics: ElectronicModel_ | None = None) -> np.ndarray:
        """Compute force on active electronic state.

        Parameters
        ----------
        electronics : ElectronicModel_, optional
            ElectronicStates from current step.

        Returns
        -------
        np.ndarray
            Force on active electronic state.
        """

    def mode_kinetic_energy(self, direction: np.ndarray) -> np.float64:
        """Calculate kinetic energy along given momentum mode.

        Parameters
        ----------
        direction : np.ndarray
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

    def needed_gradients(self) -> list[int] | None:
        """States whose forces are needed during propagation.

        Returns
        -------
        list[int] | None
            List of state indices for which gradients are needed.
            None means all states are needed.
        """
        return None

    def needed_couplings(self) -> list[tuple[int, int]] | None:
        """Coupling pairs needed during propagation.

        Returns
        -------
        list[tuple[int, int]] | None
            List of (i, j) state pairs for which derivative couplings are needed.
            None means all couplings are needed.
        """
        return None

    def _reporting_enabled(self) -> bool:
        """Check whether progress reporting is enabled."""
        return self.report_every > 0 and self.report_file is not None

    def _report_columns(self) -> list[tuple[str, str]]:
        """Return (header, formatted_value) pairs for the current report line.

        Subclasses can override to append additional columns.
        """
        assert self.electronics is not None
        temp = float(2 * self.kinetic_energy() /
                     (boltzmann * self.model.ndof))
        self._temp_buffer.append(temp)
        avg_temp = sum(self._temp_buffer) / len(self._temp_buffer)

        elapsed = time.monotonic() - self._last_report_time
        self._last_report_time = time.monotonic()

        return [
            ("Step", f"{self.nsteps:>10d}"),
            ("Time (fs)", f"{self.time / fs_to_au:>12.2f}"),
            ("Total Energy (H)", f"{self.total_energy():>17.10f}"),
            ("Avg Temp (K)", f"{avg_temp:>12.1f}"),
            ("Wall (min)", f"{elapsed / 60.0:>10.2f}"),
        ]

    def _print_report_header(self, fh: Any) -> None:
        """Print column headers to stdout and report file."""
        columns = self._report_columns()
        widths = [max(len(h), len(v)) for h, v in columns]
        header = "  ".join(f"{h:>{w}}" for (h, _), w in zip(columns, widths))
        sep = "  ".join("-" * w for w in widths)
        print(header)
        print(sep)
        fh.write(header + "\n")
        fh.write(sep + "\n")
        fh.flush()

    def _print_report(self, fh: Any) -> None:
        """Print one report line to stdout and report file."""
        columns = self._report_columns()
        line = "  ".join(v for _, v in columns)
        print(line)
        fh.write(line + "\n")
        fh.flush()

    def simulate(self) -> Trace_:
        """Run the simulation.

        Returns
        -------
        Trace_
            Trace of trajectory.
        """
        if not self.continue_simulating():
            return self.tracer

        if self.electronics is None:
            self.electronics = self.model.update(
                self.position,
                gradients=self.needed_gradients(),
                couplings=self.needed_couplings())

        if not self.restarting:
            self.trace()

        # set up progress reporting
        reporting = self._reporting_enabled()
        report_fh = None
        if reporting:
            self._temp_buffer.clear()
            self._last_report_time = time.monotonic()
            assert self.report_file is not None
            report_fh = open(self.report_file, "w")  # noqa: SIM115
            self._print_report_header(report_fh)

        # propagation
        try:
            while True:
                self.propagator(self, 1)  # pylint: disable=not-callable

                if reporting and report_fh is not None \
                        and self.nsteps % self.report_every == 0:
                    self._print_report(report_fh)

                # ending condition
                if not self.continue_simulating():
                    break

                self.trace()

            self.trace(force=True)

            if reporting and report_fh is not None:
                self._print_report(report_fh)
        finally:
            if report_fh is not None:
                report_fh.close()

        return self.tracer
