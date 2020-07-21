#!/usr/bin/env python
## @package fssh
#  Module responsible for propagating surface hopping trajectories

from __future__ import print_function, division

from .version import __version__

import copy as cp
import numpy as np

from .trajectory_sh import TrajectorySH

from typing import Any
from .typing import ArrayLike, DtypeLike, ElectronicT

## Ehrenfest dynamics
class Ehrenfest(TrajectorySH):
    def __init__(self, *args: Any, **kwargs: Any):
        TrajectorySH.__init__(self, *args, **kwargs)

    ## Ehrenfest potential energy = tr(rho * H)
    # @param electronics ElectronicStates from current step
    def potential_energy(self, electronics: ElectronicT = None) -> DtypeLike:
        if electronics is None:
            electronics = self.electronics
        return np.real(np.trace(np.dot(self.rho, electronics.hamiltonian)))

    ## Ehrenfest force = tr(rho * H')
    # @param electronics ElectronicStates from current step
    def force(self, electronics: ElectronicT = None) -> ArrayLike:
        if electronics is None:
            electronics = self.electronics
        return np.dot(np.real(np.diag(self.rho)), electronics.force)

    ## Ehrenfest never hops
    # @param electronics ElectronicStates from current step (not used)
    def surface_hopping(self, last_electronics: ElectronicT, this_electronics: ElectronicT) -> DtypeLike:
        return 0.0

