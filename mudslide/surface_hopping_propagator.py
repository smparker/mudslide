# -*- coding: utf-8 -*-
"""Propagate FSSH trajectory"""

from typing import List, Dict, Union, Any
import copy as cp

import numpy as np

from .typing import ElectronicT, ArrayLike, DtypeLike

from .util import is_string
from .propagation import propagate_exponential, propagate_interpolated_rk4
from .tracer import Trace
from .math import poisson_prob_scale
from .propagator import Propagator_

class SHVVPropagator(Propagator_):
    """Velocity Verlet propagator for surface hopping dynamics.

    This class implements the Velocity Verlet algorithm for propagating
    classical trajectories in surface hopping molecular dynamics simulations.
    """

    def __init__(self, **options: Any) -> None:
        """Constructor

        Parameters
        ----------
        **options : Any
            Option dictionary for configuration.
        """
        super().__init__()

    def __call__(self, traj: 'SurfaceHoppingMD', nsteps: int) -> None:
        """Propagate trajectory using Velocity Verlet algorithm.

        Parameters
        ----------
        traj : SurfaceHoppingMD
            Trajectory object to propagate
        nsteps : int
            Number of steps to propagate
        """
        dt = traj.dt
        # first update nuclear coordinates
        for _ in range(nsteps):
            # Advance position using Velocity Verlet
            acceleration = traj._force(traj.electronics) / traj.mass
            traj.last_position = traj.position
            traj.position += traj.velocity * dt + 0.5 * acceleration * dt * dt

            # calculate electronics at new position
            traj.last_electronics, traj.electronics = traj.electronics, traj.model.update(
                traj.position, electronics=traj.electronics)

            # Update velocity using Velocity Verlet
            last_acceleration = traj._force(traj.last_electronics) / traj.mass
            this_acceleration = traj._force(traj.electronics) / traj.mass
            traj.last_velocity = traj.velocity
            traj.velocity += 0.5 * (last_acceleration + this_acceleration) * dt

            # now propagate the electronic wavefunction to the new time
            traj.propagate_electronics(traj.last_electronics, traj.electronics, dt)
            traj.surface_hopping(traj.last_electronics, traj.electronics)

            traj.time += dt
            traj.nsteps += 1

class SHPropagator(Propagator_):
    """Surface Hopping propagator factory.

    This class serves as a factory for creating different types of propagators
    used in surface hopping molecular dynamics simulations.
    """

    def __new__(cls, model: Any, prop_options: Any = "vv") -> 'SHPropagator':
        """Create a new surface hopping propagator instance.

        Parameters
        ----------
        model : Any
            Model object defining the problem
        prop_options : Any, optional
            Propagator options, can be a string or dictionary, by default "vv"

        Returns
        -------
        SHPropagator
            A new propagator instance

        Raises
        ------
        ValueError
            If the propagator type is unknown
        """
        if is_string(prop_options):
            prop_options = {"type": prop_options}
        elif not isinstance(prop_options, dict):
            raise Exception("prop_options must be a string or a dictionary")

        proptype = prop_options.get("type", "vv")
        if proptype.lower() == "vv":
            return SHVVPropagator(**prop_options)
        else:
            raise ValueError(f"Unrecognized surface hopping propagator type: {proptype}.")
