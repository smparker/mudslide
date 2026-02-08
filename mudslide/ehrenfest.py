# -*- coding: utf-8 -*-
"""Propagating Ehrenfest trajectories"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from .surface_hopping_md import SurfaceHoppingMD

if TYPE_CHECKING:
    from .models.electronics import ElectronicModel_


class Ehrenfest(SurfaceHoppingMD):
    """Ehrenfest dynamics.

    This class implements Ehrenfest dynamics, where the electronic degrees of freedom
    are treated quantum mechanically and the nuclear degrees of freedom are treated
    classically. The force on the nuclei is computed as the expectation value of the
    force operator over the electronic density matrix.

    Surface hopping options (hopping_probability, hopping_method, forced_hop_threshold,
    zeta_list) are not used in Ehrenfest dynamics.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize Ehrenfest dynamics.

        Parameters
        ----------
        *args : Any
            Positional arguments passed to SurfaceHoppingMD
        **kwargs : Any
            Keyword arguments passed to SurfaceHoppingMD
        """
        kwargs.setdefault("outcome_type", "populations")
        SurfaceHoppingMD.__init__(self, *args, **kwargs)

    def needed_gradients(self) -> list[int] | None:
        """Ehrenfest needs all forces since it sums over all states.

        Returns
        -------
        None
            None means all state gradients are needed.
        """
        return None

    def potential_energy(self,
                         electronics: ElectronicModel_ | None = None) -> float:
        """Calculate Ehrenfest potential energy.

        The potential energy is computed as the trace of the product of the
        density matrix and the Hamiltonian.

        Parameters
        ----------
        electronics : ElectronicModel, optional
            ElectronicStates from current step, by default None

        Returns
        -------
        np.floating
            Potential energy
        """
        if electronics is None:
            electronics = self.electronics
        assert electronics is not None
        return np.real(np.trace(np.dot(self.rho, electronics.hamiltonian)))

    def force(self, electronics: ElectronicModel_ | None = None) -> np.ndarray:
        """Calculate Ehrenfest force.

        The force is computed as the trace of the product of the density matrix
        and the force operator.

        Parameters
        ----------
        electronics : ElectronicModel, optional
            ElectronicStates from current step, by default None

        Returns
        -------
        ArrayLike
            Force vector
        """
        if electronics is None:
            electronics = self.electronics
        assert electronics is not None

        out = np.zeros([electronics.ndof])
        for i in range(electronics.nstates):
            out += np.real(self.rho[i, i]) * electronics.force(i)
        return out

    def surface_hopping(self, last_electronics: ElectronicModel_,
                        this_electronics: ElectronicModel_) -> None:
        """Handle surface hopping.

        In Ehrenfest dynamics, surface hopping is not performed as the electronic
        states are treated quantum mechanically.

        Parameters
        ----------
        last_electronics : ElectronicModel
            Electronic states from previous step
        this_electronics : ElectronicModel
            Electronic states from current step
        """
        return
