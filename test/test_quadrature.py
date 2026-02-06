#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit testing for mudslide"""

import mudslide
import numpy as np


def _setup_quadrature():
    """Setup function"""
    a = 0
    b = 1
    n = 11

    evenx = [i / (n - 1) for i in range(n)]
    flatw = [
        0.09090909, 0.09090909, 0.09090909, 0.09090909, 0.09090909, 0.09090909, 0.09090909, 0.09090909, 0.09090909,
        0.09090909, 0.09090909
    ]
    return a, b, n, evenx, flatw


def test_midpoint_quadrature():
    """Midpoint quadrature"""
    a, b, n, evenx, flatw = _setup_quadrature()
    refx = [
        0.04545455, 0.13636364, 0.22727273, 0.31818182, 0.40909091, 0.5, 0.59090909, 0.68181818, 0.77272727,
        0.86363636, 0.95454545
    ]
    refw = flatw
    x, w = mudslide.integration.quadrature(n, a, b, "midpoint")
    assert len(x) == n
    assert np.all(np.isclose(x, refx))

    assert len(w) == n
    assert np.all(np.isclose(w, refw))


def test_trapezoid_quadrature():
    """Trapezoid quadrature"""
    a, b, n, evenx, flatw = _setup_quadrature()
    refx = evenx
    refw = [0.1 for i in range(n)]
    refw[0] *= 0.5
    refw[-1] *= 0.5

    x, w = mudslide.integration.quadrature(n, a, b, "trapezoid")
    assert len(x) == n
    assert np.all(np.isclose(x, refx))

    assert len(w) == n
    assert np.all(np.isclose(w, refw))


def test_simpson_quadrature():
    """Simpson quadrature"""
    a, b, n, evenx, flatw = _setup_quadrature()
    refx = evenx
    refw = [
        0.03333333, 0.13333333, 0.06666667, 0.13333333, 0.06666667, 0.13333333, 0.06666667, 0.13333333, 0.06666667,
        0.13333333, 0.03333333
    ]

    x, w = mudslide.integration.quadrature(n, a, b, "simpson")
    assert len(x) == n
    assert np.all(np.isclose(x, refx))

    assert len(w) == n
    assert np.all(np.isclose(w, refw))


def test_clenshawcurtis_quadrature():
    """Clenshaw-Curtis quadrature"""
    a, b, n, evenx, flatw = _setup_quadrature()
    refx = [
        0., 0.02447174, 0.0954915, 0.20610737, 0.3454915, 0.5, 0.6545085, 0.79389263, 0.9045085, 0.97552826, 1.
    ]
    refw = [
        0.00505051, 0.04728953, 0.09281761, 0.12679417, 0.14960664, 0.15688312, 0.14960664, 0.12679417, 0.09281761,
        0.04728953, 0.00505051
    ]

    x, w = mudslide.integration.quadrature(n, a, b, "cc")
    assert len(x) == n
    assert np.all(np.isclose(x, refx))

    assert len(w) == n
    assert np.all(np.isclose(w, refw))


def test_gasslegendre_quadrature():
    """Gauss-Legendre quadrature"""
    a, b, n, evenx, flatw = _setup_quadrature()
    refx = [
        0.01088567, 0.0564687, 0.134924, 0.24045194, 0.36522842, 0.5, 0.63477158, 0.75954806, 0.865076, 0.9435313,
        0.98911433
    ]
    refw = [
        0.02783428, 0.06279018, 0.09314511, 0.11659688, 0.13140227, 0.13646254, 0.13140227, 0.11659688, 0.09314511,
        0.06279018, 0.02783428
    ]

    x, w = mudslide.integration.quadrature(n, a, b, "gl")
    assert len(x) == n
    assert np.all(np.isclose(x, refx))

    assert len(w) == n
    assert np.all(np.isclose(w, refw))
