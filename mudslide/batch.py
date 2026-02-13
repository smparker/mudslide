# -*- coding: utf-8 -*-
"""Code for running batches of trajectories."""

from __future__ import annotations

from typing import Any, Iterator, Tuple, TYPE_CHECKING
import logging
import queue

import numpy as np
from numpy.typing import ArrayLike

from .constants import boltzmann
from .tracer import TraceManager

if TYPE_CHECKING:
    from .models.electronics import ElectronicModel_

logger = logging.getLogger("mudslide")


class TrajGenConst:
    """Generator for static initial conditions.

    Parameters
    ----------
    position : np.ndarray
        Initial position.
    velocity : np.ndarray
        Initial velocity.
    initial_state : Any
        Initial state specification, should be either an integer or "ground".
    seed : Any, optional
        Entropy seed for random generator (unused), by default None.

    Yields
    ------
    Iterator
        Generator yielding tuples of (position, velocity, initial_state, params).
    """

    def __init__(self,
                 position: np.ndarray,
                 velocity: np.ndarray,
                 initial_state: Any,
                 seed: Any = None):
        self.position = position
        self.velocity = velocity
        self.initial_state = initial_state
        self.seed_sequence = np.random.SeedSequence(seed)

    def __call__(self, nsamples: int) -> Iterator:
        """Generate initial conditions.

        Parameters
        ----------
        nsamples : int
            Number of initial conditions to generate.

        Yields
        ------
        Iterator
            Generator yielding tuples of (position, velocity, initial_state, params).
        """
        seedseqs = self.seed_sequence.spawn(nsamples)
        for i in range(nsamples):
            yield (self.position, self.velocity, self.initial_state, {
                "seed_sequence": seedseqs[i]
            })


class TrajGenNormal:
    """Generator for normally distributed initial conditions.

    Parameters
    ----------
    position : np.ndarray
        Center of normal distribution for position.
    velocity : np.ndarray
        Center of normal distribution for velocity.
    initial_state : Any
        Initial state designation.
    sigma : np.ndarray
        Standard deviation of distribution.
    seed : Any, optional
        Initial seed to give to trajectory, by default None.
    seed_traj : Any, optional
        Seed for trajectory generation, by default None.
    """

    def __init__(self,
                 position: np.ndarray,
                 velocity: np.ndarray,
                 initial_state: Any,
                 sigma: np.ndarray,
                 seed: Any = None,
                 seed_traj: Any = None):
        self.position = position
        self.position_deviation = 0.5 * sigma
        self.velocity = velocity
        self.velocity_deviation = 1.0 / sigma
        self.initial_state = initial_state
        self.seed_sequence = np.random.SeedSequence(seed)
        self.random_state = np.random.default_rng(seed_traj)

    def vskip(self, vtest: np.ndarray) -> bool:
        """Determine whether to skip given velocity.

        Parameters
        ----------
        vtest : np.ndarray
            Velocity to test.

        Returns
        -------
        bool
            True if velocity should be skipped, False otherwise.
        """
        return bool(np.any(vtest < 0.0))

    def __call__(self, nsamples: int) -> Iterator:
        """Generate initial conditions.

        Parameters
        ----------
        nsamples : int
            Number of initial conditions requested.

        Yields
        ------
        Iterator
            Generator yielding tuples of (position, velocity, initial_state, params).
        """
        seedseqs = self.seed_sequence.spawn(nsamples)
        for i in range(nsamples):
            x = self.random_state.normal(self.position, self.position_deviation)
            v = self.random_state.normal(self.velocity, self.velocity_deviation)

            if self.vskip(v):
                continue
            yield (x, v, self.initial_state, {"seed_sequence": seedseqs[i]})


class TrajGenBoltzmann:
    """Generate velocities randomly according to the Boltzmann distribution.

    Parameters
    ----------
    position : np.ndarray
        Initial positions.
    mass : np.ndarray
        Array of particle masses.
    temperature : float
        Initial temperature to determine velocities.
    initial_state : Any
        Initial state designation.
    scale : bool, optional
        Whether to scale velocities, by default True.
    seed : Any, optional
        Initial seed to give trajectory, by default None.
    velocity_seed : Any, optional
        Initial seed for velocity random generator, by default None.
    """

    def __init__(self,
                 position: np.ndarray,
                 mass: np.ndarray,
                 temperature: float,
                 initial_state: Any,
                 scale: bool = True,
                 seed: Any = None,
                 velocity_seed: Any = None):
        assert position.shape == mass.shape

        self.position = position
        self.mass = mass
        self.temperature = temperature
        self.initial_state = initial_state
        self.scale = scale
        self.seed_sequence = np.random.SeedSequence(seed)
        self.random_state = np.random.default_rng(velocity_seed)
        self.kt = boltzmann * self.temperature
        self.sigma = np.sqrt(self.kt * self.mass)

    def __call__(self, nsamples: int) -> Iterator:
        """Generate initial conditions.

        Parameters
        ----------
        nsamples : int
            Number of initial conditions requested.

        Yields
        ------
        Iterator
            Generator yielding tuples of (position, velocity, initial_state, params).
        """
        seedseqs = self.seed_sequence.spawn(nsamples)
        for i in range(nsamples):
            x = self.position
            p = self.random_state.normal(0.0, self.sigma)
            v = p / self.mass

            if self.scale:
                avg_KE = 0.5 * np.dot(v**2, self.mass) / x.size
                kbT2 = 0.5 * self.kt
                scal = np.sqrt(kbT2 / avg_KE)
                v *= scal

            yield (x, v, self.initial_state, {"seed_sequence": seedseqs[i]})


class BatchedTraj:
    """Class to manage many SurfaceHoppingMD trajectories.

    Requires a model object which is a class that has functions V(x), dV(x), nstates, and ndof
    that return the Hamiltonian at position x, gradient of the Hamiltonian at position x,
    number of electronic states, and dimension of nuclear space, respectively.

    Parameters
    ----------
    model : ElectronicModel
        Object used to describe the model system.
    traj_gen : Any
        Generator object to generate initial conditions.
    trajectory_type : Any
        Surface hopping trajectory class.
    tracemanager : Any, optional
        Object to collect results, by default None.
    **inp : Any
        Additional input options.

    Notes
    -----
    Accepted keyword arguments and their defaults:
    | key                |   default                  |
    ---------------------|----------------------------|
    | t0                 | 0.0                        |
    | seed               | None (date)                |
    """

    batch_only_options = ["samples"]

    def __init__(self,
                 model: ElectronicModel_,
                 traj_gen: Any,
                 trajectory_type: Any,
                 tracemanager: Any = None,
                 **inp: Any):
        self.model = model
        if tracemanager is None:
            self.tracemanager = TraceManager()
        else:
            self.tracemanager = tracemanager
        self.trajectory = trajectory_type
        self.traj_gen = traj_gen
        self.batch_options = {}

        # statistical parameters
        self.batch_options["samples"] = inp.get("samples", 2000)

        # other options get copied over
        self.traj_options = {}
        for x in inp:
            if x not in self.batch_only_options:
                self.traj_options[x] = inp[x]

    def compute(self) -> TraceManager:
        """Run batch of trajectories and return aggregate results.

        Returns
        -------
        TraceManager
            Object containing the results.
        """
        nsamples = self.batch_options["samples"]

        traj_queue: Any = queue.Queue()

        for x0, v0, initial, params in self.traj_gen(nsamples):
            traj_input = self.traj_options
            traj_input.update(params)
            traj = self.trajectory(self.model,
                                   x0,
                                   v0,
                                   initial,
                                   self.tracemanager.spawn_tracer(),
                                   queue=traj_queue,
                                   **traj_input)
            traj_queue.put(traj)

        while not traj_queue.empty():
            traj = traj_queue.get()
            results = traj.simulate()
            self.tracemanager.merge_tracer(results)

        self.tracemanager.outcomes = self.tracemanager.outcome()
        return self.tracemanager
