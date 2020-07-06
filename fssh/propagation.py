#!/usr/bin/env python
## @package propagation
#  Module responsible for propagating ODEs encountered in quantum dynamics

from __future__ import print_function, division

import numpy as np

def rk4(y0, ydot, t0, tf, nsteps):
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
