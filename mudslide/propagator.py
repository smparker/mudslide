# -*- coding: utf-8 -*-
"""Propagators for ODEs from quantum dynamics"""

from typing import Any


class Propagator_:
    """Base class for propagators.

    This class serves as the base class for all propagators in the system.
    It defines the interface that all propagator implementations must follow.
    """

    def __call__(self, traj: Any, nsteps: int) -> None:
        """Propagate a trajectory for a specified number of steps.

        Parameters
        ----------
        traj : Any
            The trajectory object to propagate
        nsteps : int
            Number of steps to propagate

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses
        """
        raise NotImplementedError
