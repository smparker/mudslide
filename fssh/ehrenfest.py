#!/usr/bin/env python
## @package fssh
#  Module responsible for propagating surface hopping trajectories

from __future__ import print_function, division

from .version import __version__

import copy as cp
import numpy as np

from .trajectory_sh import TrajectorySH

## Ehrenfest dynamics
class Ehrenfest(TrajectorySH):
    def __init__(self, *args, **kwargs):
        TrajectorySH.__init__(self, *args, **kwargs)

    ## Ehrenfest potential energy = tr(rho * H)
    # @param electronics ElectronicStates from current step
    def potential_energy(self, electronics):
        return np.real(np.trace(np.dot(self.rho, electronics.hamiltonian)))

    ## Ehrenfest force = tr(rho * H')
    # @param electronics ElectronicStates from current step
    def force(self, electronics):
        return np.dot(np.real(np.diag(self.rho)), electronics.force)

    ## Ehrenfest never hops
    # @param electronics ElectronicStates from current step (not used)
    def surface_hopping(self, electronics):
        return 0.0

