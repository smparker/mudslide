# -*- coding: utf-8 -*-
"""Propagating Ehrenfest trajectories"""

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .surface_hopping_md import SurfaceHoppingMD


class Ehrenfest(SurfaceHoppingMD):
    """Ehrenfest dynamics.

    This class implements Ehrenfest dynamics, where the electronic degrees of freedom
    are treated quantum mechanically and the nuclear degrees of freedom are treated
    classically. The force on the nuclei is computed as the expectation value of the
    force operator over the electronic density matrix.
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
        SurfaceHoppingMD.__init__(self, *args, **kwargs)

    def potential_energy(self, electronics: 'ElectronicModel_' = None) -> np.floating:
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
        return np.real(np.trace(np.dot(self.rho, electronics.hamiltonian)))

    def _force(self, electronics: 'ElectronicModel_' = None) -> ArrayLike:
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

        out = np.zeros([electronics.ndof])
        for i in range(electronics.nstates):
            out += np.real(self.rho[i,i]) * electronics.force(i)
        return out

    def surface_hopping(self, last_electronics: 'ElectronicModel_',
                        this_electronics: 'ElectronicModel_'):
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
