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
    """Surface Hopping Velocity Verlet propagator"""

    def __init__(self, **options: Any) -> None:
        """Constructor

        :param options: option dictionary
        """
        super().__init__()

    def __call__(self, traj: 'SurfaceHoppingMD', nsteps: int) -> None:
        """Propagate trajectory using Surface Hopping Velocity Verlet algorithm

        :param traj: trajectory object to propagate
        :param nsteps: number of steps to propagate
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
    """Surface Hopping propagator factory"""

    def __new__(cls, model: Any, prop_options: Any = "vv") -> 'SHPropagator':
        """Factory method to create a Surface Hopping propagator

        :param model: Model object defining problem
        :param prop_options: options for propagator, can be "vv" or "fssh"
        :return: SHPropagator object
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
