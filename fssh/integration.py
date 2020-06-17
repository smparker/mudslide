#!/usr/bin/env python
## @package integration
#  Module responsible for propagating surface hopping trajectories

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

def clenshaw_curtis(n, a=-1.0, b=1.0):
    """
    Computes the points and weights for a Clenshaw-Curtis integration
    from a to b. In other words, for the approximation to the integral

    \int_a^b f(x) dx \approx \sum_{i=0}^{n} w_i f(x_i)

    with the Clenshaw-Curtis quadrature, this function returns the
    positions x_i and the weights w_i.
    """
    assert b > a and n > 1

    theta = np.pi * np.flip(np.arange(n+1)) / n
    xx = np.cos(theta) * 0.5 * (b - a) + 0.5 * (a + b)

    wcc0 = 1.0/(n*n - 1 + (n%2))

    # build v vector
    v = np.zeros(n)
    v[:n//2] = 2.0/(1.0 - 4.0 * np.arange(n//2)**2)
    v[n//2] = (n - 3) / (2 * (n//2) - 1) - 1
    for k in range(1, (n+1)//2):
        v[n-k] = np.conj(v[k])

    # build g vector
    g = np.zeros(n)
    g[:n//2] = -wcc0
    g[n//2] = wcc0 * ( (2 - (n%2)) * n - 1 )
    for k in range(1, (n+1)//2):
        g[n-k] = np.conj(g[k])

    h = v + g
    wcc = np.fft.ifft(h)

    # sanity check
    imag_norm = np.linalg.norm(np.imag(wcc))
    assert imag_norm < 1e-14

    out = np.zeros(n+1)
    out[:n] = np.real(wcc)
    out[n] = out[0]
    out = np.flip(out) # might be redundant, but for good measure
    out *= 0.5 * (b - a)

    return xx, out

def midpoint(n, a=-1.0, b=1.0):
    """
    Returns the points and weights for a midpoint integration
    from a to b. In other words, for the approximation to the integral

    \int_a^b f(x) dx \approx \frac{b-a}{n} \sum_{i=0}^n f((x_0 + x_1)/2)
    """
    assert b > a and n > 1

    weights = np.ones(n) * (b - a) / n
    points = np.zeros(n)
    for i in range(n):
        points[i] = a + ((b - a) / n) * (i + 0.5)

    return points, weights

def trapezoid(n, a=-1.0, b=1.0):
    """
    Returns the points and weights for a trapezoid integration
    from a to b. In other words, for the approximation to the integral

    \int_a^b f(x) dx \approx \frac{b-a}{n} \sum_{i=0}^n f((x_0 + x_1)/2)
    """
    assert b > a and n > 1

    ninterval = n - 1

    weights = np.ones(n) * (b - a) / ninterval
    weights[0] *= 0.5
    weights[-1] *= 0.5
    points = np.zeros(n)
    for i in range(n):
        points[i] = a + ((b - a) / ninterval) * i

    return points, weights

def quadrature(n, a=-1.0, b=1.0, method="gl"):
    """
    Returns a quadrature rule for the specified method and bounds
    """
    if method.lower() == "cc" or method.lower() == "clenshaw-curtis":
        return clenshaw_curtis(n, a, b)
    elif method.lower() == "gl" or method.lower() == "gauss-legendre":
        points, weights = np.polynomial.legendre.leggauss(n)
        points = points * 0.5 * (b - a) + 0.5 * (a + b)
        weights *= 0.5
        return points, weights
    elif method.lower() == "midpoint" or method.lower() == "mp":
        return midpoint(n, a, b)
    elif method.lower() == "trapezoid":
        return trapezoid(n, a, b)
    else:
        raise Exception("Unrecognized quadrature choice")
