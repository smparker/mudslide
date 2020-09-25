# -*- coding: utf-8 -*-
"""Propagators for ODEs from quantum dynamics"""

from __future__ import division

import numpy as np

from typing import Callable
from .typing import ArrayLike, DtypeLike

def rk4(y0: ArrayLike, ydot: Callable, t0: DtypeLike, tf: DtypeLike, nsteps: int) -> ArrayLike:
    """
    Propagates using 4th-order Runge-Kutta (RK4).
    """
    dt = (tf - t0)/nsteps

    y = np.copy(y0)

    for i in range(nsteps):
        t = t0 + i*dt

        k1 = ydot(y, t)
        k2 = ydot(y + 0.5*dt*k1, t + 0.5*dt)
        k3 = ydot(y + 0.5*dt*k2, t + 0.5*dt)
        k4 = ydot(y + dt*k3, t + dt)

        y += (dt/6.0) * ( k1 + 2.0 * k2 + 2.0 * k3 + k4 )

    return y
