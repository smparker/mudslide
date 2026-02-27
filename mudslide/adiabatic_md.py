# -*- coding: utf-8 -*-
"""Propagate Adiabatic MD trajectory.

This module provides functionality for propagating adiabatic molecular dynamics trajectories,
similar to ground state molecular dynamics simulations.
"""

from __future__ import annotations

from typing import Dict, Any, TYPE_CHECKING

import numpy as np

from .trajectory_md import TrajectoryMD
from .propagator import Propagator_
from .adiabatic_propagator import AdiabaticPropagator

if TYPE_CHECKING:
    from .models.electronics import ElectronicModel_


class AdiabaticMD(TrajectoryMD):
    """Class to propagate a single adiabatic trajectory, like ground state MD.

    This class handles the propagation of molecular dynamics trajectories in the
    adiabatic regime, similar to ground state molecular dynamics.
    """
    def make_propagator(self, model: ElectronicModel_,
                        options: Dict[str, Any]) -> Propagator_:
        """Create the adiabatic propagator.

        Parameters
        ----------
        model : Any
            Model object defining problem.
        options : Dict[str, Any]
            Options dictionary.

        Returns
        -------
        Propagator_
            Adiabatic propagator instance.
        """
        return AdiabaticPropagator(model, options.get("propagator", "VV"))  # type: ignore[return-value]

    @classmethod
    def restart(cls, model: ElectronicModel_, log: Any, **options: Any) -> 'AdiabaticMD':
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
        if electronics is None:
            electronics = self.electronics
        assert electronics is not None
        return electronics.energies[0]

    def force(self, electronics: ElectronicModel_ | None = None) -> np.ndarray:
        """Compute force on ground state.

        Parameters
        ----------
        electronics : ElectronicModel_, optional
            ElectronicStates from current step.

        Returns
        -------
        np.ndarray
            Force on ground electronic state.
        """
        if electronics is None:
            electronics = self.electronics
        assert electronics is not None
        return electronics.force(0)
