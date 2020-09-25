# -*- coding: utf-8 -*-
"""Propagating Ehrenfest trajectories"""

import numpy as np

from .trajectory_sh import TrajectorySH

from typing import Any
from .typing import ArrayLike, DtypeLike, ElectronicT

class Ehrenfest(TrajectorySH):
    """Ehrenfest dynamics"""
    def __init__(self, *args: Any, **kwargs: Any):
        TrajectorySH.__init__(self, *args, **kwargs)

    def potential_energy(self, electronics: ElectronicT = None) -> DtypeLike:
        """Ehrenfest potential energy = tr(rho * H)

        :param electronics: ElectronicStates from current step
        """
        if electronics is None:
            electronics = self.electronics
        return np.real(np.trace(np.dot(self.rho, electronics.hamiltonian)))

    def force(self, electronics: ElectronicT = None) -> ArrayLike:
        """Ehrenfest potential energy = tr(rho * H')

        :param electronics: ElectronicStates from current step
        """
        if electronics is None:
            electronics = self.electronics
        return np.dot(np.real(np.diag(self.rho)), electronics.force)

    def surface_hopping(self, last_electronics: ElectronicT, this_electronics: ElectronicT):
        """Ehrenfest never hops"""
        return

