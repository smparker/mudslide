# -*- coding: utf-8 -*-
"""Propagating Ehrenfest trajectories"""

import numpy as np

from .surface_hopping_md import SurfaceHoppingMD

from typing import Any
from .typing import ArrayLike, DtypeLike, ElectronicT


class Ehrenfest(SurfaceHoppingMD):
    """Ehrenfest dynamics"""

    def __init__(self, *args: Any, **kwargs: Any):
        SurfaceHoppingMD.__init__(self, *args, **kwargs)

    def potential_energy(self, electronics: ElectronicT = None) -> DtypeLike:
        """Ehrenfest potential energy = tr(rho * H)

        :param electronics: ElectronicStates from current step
        """
        if electronics is None:
            electronics = self.electronics
        return np.real(np.trace(np.dot(self.rho, electronics.hamiltonian())))

    def _force(self, electronics: ElectronicT = None) -> ArrayLike:
        """Ehrenfest potential energy = tr(rho * H')

        :param electronics: ElectronicStates from current step
        """
        if electronics is None:
            electronics = self.electronics

        out = np.zeros([electronics.ndim()])
        for i in range(electronics.nstates()):
            out += np.real(self.rho[i,i]) * electronics.force(i)
        return out

    def surface_hopping(self, last_electronics: ElectronicT, this_electronics: ElectronicT):
        """Ehrenfest never hops"""
        return
