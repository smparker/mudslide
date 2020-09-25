# -*- coding: utf-8 -*-
"""Quadrature implementations"""

from __future__ import division

import numpy as np

from typing import Tuple
from .typing import ArrayLike

def clenshaw_curtis(n: int, a: float = -1.0, b: float = 1.0) -> Tuple[ArrayLike,ArrayLike]:
    """
    Computes the points and weights for a Clenshaw-Curtis integration
    from a to b. In other words, for the approximation to the integral

    \int_a^b f(x) dx \approx \sum_{i=0}^{n} w_i f(x_i)

    with the Clenshaw-Curtis quadrature, this function returns the
    positions x_i and the weights w_i.
    """
    assert b > a and n > 1

    npoints = n
    nsegments = n - 1
    theta = np.pi * np.flip(np.arange(npoints)) / nsegments
    xx = np.cos(theta) * 0.5 * (b - a) + 0.5 * (a + b)

    wcc0 = 1.0/(nsegments*nsegments - 1 + (nsegments%2))

    # build v vector
    v = np.zeros(nsegments)
    v[:nsegments//2] = 2.0/(1.0 - 4.0 * np.arange(nsegments//2)**2)
    v[nsegments//2] = (nsegments - 3) / (2 * (nsegments//2) - 1) - 1

    kk = np.arange(1, npoints//2)
    v[nsegments-kk] = np.conj(v[kk])

    # build g vector
    g = np.zeros(nsegments)
    g[:nsegments//2] = -wcc0
    g[nsegments//2] = wcc0 * ( (2 - (nsegments%2)) * nsegments - 1 )
    g[nsegments-kk] = np.conj(g[kk])

    h = v + g
    wcc = np.fft.ifft(h)

    # sanity check
    imag_norm = np.linalg.norm(np.imag(wcc))
    assert imag_norm < 1e-14

    out = np.zeros(npoints)
    out[:nsegments] = np.real(wcc)
    out[nsegments] = out[0]
    out = np.flip(out) # might be redundant, but for good measure
    out *= 0.5 * (b - a)

    return xx, out

def midpoint(n: int, a: float = -1.0, b: float = 1.0) -> Tuple[ArrayLike,ArrayLike]:
    """
    Returns the points and weights for a midpoint integration
    from a to b. In other words, for the approximation to the integral

    \int_a^b f(x) dx \approx \frac{b-a}{n} \sum_{i=0}^n f((x_0 + x_1)/2)
    """
    assert b > a and n > 1

    weights = np.ones(n) * (b - a) / n
    points = a + ((b - a) / n * (np.arange(n) + 0.5))

    return points, weights

def trapezoid(n: int, a: float = -1.0, b: float = 1.0) -> Tuple[ArrayLike,ArrayLike]:
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
    points = a + ((b - a) / ninterval) * np.arange(n)

    return points, weights

def simpson(n: int, a: float = -1.0, b: float = 1.0) -> Tuple[ArrayLike,ArrayLike]:
    """
    Returns the points and weights for a simpson rule integration
    from a to b. In other words, for the approximation to the integral

    \int_a^b f(x) dx \approx \frac{b-a}{n} \sum_{i=0}^n f((x_0 + x_1)/2)
    """
    assert b > a and n > 1

    if n%2 != 1:
        raise Exception("Simpson's rule must be defined with an odd number of points (even number of intervals)")

    ninterval = n - 1

    weights = np.ones(n)
    for i in range(1, ninterval-1, 2):
        weights[i] = 4.0
        weights[i+1] = 2.0
    weights[ninterval-1] = 4.0
    weights *= (b-a) / ninterval / 3.0

    points = a + ((b - a) / ninterval) * np.arange(n)

    return points, weights

def quadrature(n: int, a: float = -1.0, b: float = 1.0, method: str = "gl") -> Tuple[ArrayLike,ArrayLike]:
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
    elif method.lower() == "simpson":
        return simpson(n, a, b)
    else:
        raise Exception("Unrecognized quadrature choice")
