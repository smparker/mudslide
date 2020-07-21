#!/usr/bin/env python
## @package fssh
#  Module responsible for propagating surface hopping trajectories

from __future__ import print_function, division

from .version import __version__

import copy as cp
import numpy as np

from .trajectory_sh import TrajectorySH

from typing import Any, List, Dict
from .typing import ArrayLike

## Trajectory surface hopping using a cumulative approach rather than instantaneous
#
#  Instead of using a random number generator at every time step to test for a hop,
#  hops occur when the cumulative probability of a hop crosses a randomly determined
#  threshold. Swarmed results should be identical to the traditional variety, but
#  should be a bit easier to reproduce since far fewer random numbers are ever needed.
class TrajectoryCum(TrajectorySH):
    ## Constructor (see TrajectorySH constructor)
    def __init__(self, *args: Any, **kwargs: Any):
        TrajectorySH.__init__(self, *args, **kwargs)

        self.prob_cum = np.longdouble(0.0)
        self.zeta = self.random()

    ## returns loggable data
    def snapshot(self) -> Dict:
        out = {
            "time" : self.time,
            "position"  : np.copy(self.position),
            "momentum"  : self.mass * np.copy(self.velocity),
            "potential" : self.potential_energy(),
            "kinetic"   : self.kinetic_energy(),
            "energy"    : self.total_energy(),
            "density_matrix" : np.copy(self.rho),
            "active"    : self.state,
            "electronics" : self.electronics,
            "hopping"   : self.hopping,
            "zeta"      : self.zeta,
            "prob_cum"  : self.prob_cum
            }
        return out

    ## given a set of probabilities, determines whether and where to hop
    # @param probs [nstates] numpy array of individual hopping probabilities
    #  returns (do_hop, target_state)
    def hopper(self, probs: ArrayLike) -> List[Dict]:
        accumulated = np.longdouble(self.prob_cum)
        probs[self.state] = 0.0 # ensure self-hopping is nonsense
        gkdt = np.sum(probs)

        accumulated += (accumulated - 1.0) * np.expm1(-gkdt)
        if accumulated > self.zeta: # then hop
            # where to hop
            hop_choice = probs / gkdt

            zeta = self.zeta
            target = self.random_state.choice(list(range(self.model.nstates())), p=hop_choice)

            # reset probabilities and random
            self.prob_cum = 0.0
            self.zeta = self.random()

            return [ {"target" : target, "weight" : 1.0, "zeta" : zeta, "prob" : accumulated} ]

        self.prob_cum = accumulated
        return []

