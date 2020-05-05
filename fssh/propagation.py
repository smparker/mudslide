#!/usr/bin/env python
## @package propagation
#  Module responsible for propagating ODEs encountered in quantum dynamics

# fssh: program to run surface hopping simulations for model problems
# Copyright (C) 2018-2020, Shane Parker <shane.parker@case.edu>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
