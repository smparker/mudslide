#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit testing for larger order integrants"""

import numpy as np
import pytest
from mudslide.exceptions import ConfigurationError
from mudslide.even_sampling import SpawnStack
import itertools

def test_2D_integral():
    def two_d_poly_np(x, y):
        c = np.array(
            [
                [-12, 0, -1],
                [0, 3, 0],
                [2, 0, 0],
            ]
        )
        return np.polynomial.polynomial.polyval2d(x, y, c)


    ss = SpawnStack.from_quadrature(nsamples=[5,5])
    pnts_wghts = ss.unravel()

    results_2Dpoly = sum (
         [
             two_d_poly_np(x,y)* w for ((x, y), w) in pnts_wghts
         ]
    )

    results_analytical_2D = -131/12

    assert results_2Dpoly == pytest.approx(results_analytical_2D)

def test_3D_integral():
    def three_d_poly_np(x, y, z):
        c = np.array(
        [
            [
                [1, 2, 1],
                [2, 4, 2],
                [1, 2, 1],
            ],
            [
                [2, 4, 2],
                [4, 8, 4],
                [2, 4, 2],
            ],
            [
                [1, 2, 1],
                [2, 4, 2],
                [1, 2, 1],
            ]  ]

        )
        return np.polynomial.polynomial.polyval3d(x, y, z, c)
    ss = SpawnStack.from_quadrature(nsamples=[7,7,7])
    pnts_wghts = ss.unravel()

    results_3Dpoly = sum (
         [
             three_d_poly_np(x,y,z)* w for ((x, y, z), w) in pnts_wghts
         ]
    )

    results_analytical_3D = 343/27
    results_analytical_3D = 343/27
    assert results_3Dpoly == pytest.approx(results_analytical_3D)

def test_dimension():
    sample_stack = []
    ss = SpawnStack.from_quadrature (nsamples = [2,2,2])

    ss_2D = SpawnStack.from_quadrature (nsamples = [2,2])

    ss_1D = SpawnStack.from_quadrature (nsamples = [2])

    results_empty = [{'zeta': 1.0, 'dw': 1.0, 'children': [], 'spawn_size': 1}]
    results_append = [{'zeta': 1.0, 'dw': 1.0, 'children': [{'zeta': 0.1, 'dw': 0.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}]

    zetas = [ss_1D.sample_stack[i]["zeta"] for i in range (2)]
    dws = [ss_1D.sample_stack[i]["dw"] for i in range (2)]

    ss_2D.append_layer(zetas = zetas, dws = dws, stack = ss_2D.sample_stack)

    zetas = [0.1]
    dws = [0.1]
    assert ss_2D.sample_stack[0] == ss.sample_stack[0]
    assert ss_2D.sample_stack[1] == ss.sample_stack[1]

    merged = list(itertools.chain(*ss.sample_stack))
    merged = list(itertools.chain(*ss.sample_stack))


def test_unravel_2D():
    ss = SpawnStack.from_quadrature(nsamples=[2,2])
    ss.unravel()
    pnts_wghts = ss.unravel()

    results_unravel_2D =  [((0.21132486540518713, 0.21132486540518713), 0.25),
                            ((0.21132486540518713, 0.7886751345948129), 0.25),
                            ((0.7886751345948129, 0.21132486540518713), 0.25),
                            ((0.7886751345948129, 0.7886751345948129), 0.25)]

    assert results_unravel_2D == pnts_wghts


def test_unravel_3D():
    ss = SpawnStack.from_quadrature(nsamples=[2,2,2])
    ss.unravel()
    pnts_wghts = ss.unravel()

    results_unravel_3D = [((0.21132486540518713, 0.21132486540518713, 0.21132486540518713), 0.125),
                         ((0.21132486540518713, 0.21132486540518713, 0.7886751345948129), 0.125),
                         ((0.21132486540518713, 0.7886751345948129, 0.21132486540518713), 0.125),
                         ((0.21132486540518713, 0.7886751345948129, 0.7886751345948129), 0.125),
                         ((0.7886751345948129, 0.21132486540518713, 0.21132486540518713), 0.125),
                         ((0.7886751345948129, 0.21132486540518713, 0.7886751345948129), 0.125),
                         ((0.7886751345948129, 0.7886751345948129, 0.21132486540518713), 0.125),
                         ((0.7886751345948129, 0.7886751345948129, 0.7886751345948129), 0.125)]

    assert results_unravel_3D == pnts_wghts


def test_append_layer():
    sample_stack = []
    ss = SpawnStack(sample_stack = sample_stack)

    results_empty = [{'zeta': 1.0, 'dw': 1.0, 'children': [], 'spawn_size': 1}]
    results_append = [{'zeta': 1.0, 'dw': 1.0, 'children': [{'zeta': 0.1, 'dw': 0.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}]

    zetas = [1.0]
    dws = [1.0]
    ss.append_layer(zetas=zetas, dws=dws, stack=ss.sample_stack)

    assert results_empty[0] == ss.sample_stack[0]


    zetas = [0.1]
    dws = [0.1]
    ss.append_layer(zetas=zetas, dws=dws, stack=ss.sample_stack)

    assert results_append[0] == ss.sample_stack[0]

def test_ss_dimesion():
    ss = SpawnStack.from_quadrature(nsamples=[2, 2], mcsamples=3)
    sample_stack = ss.sample_stack

    zetas=[1.0, 1.1]
    dws=[2.0, 2.1]
    for layers in range(5):
        ss.append_layer(zetas, dws)


    results_ss_dimension= [{'zeta': 0.21132486540518713, 'dw': 0.5, 'children': [{'zeta': 0.21132486540518713, 'dw': 0.5, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 3}, {'zeta': 0.7886751345948129, 'dw': 0.5, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 3}], 'spawn_size': 1}, {'zeta': 0.7886751345948129, 'dw': 0.5, 'children': [{'zeta': 0.21132486540518713, 'dw': 0.5, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 3}, {'zeta': 0.7886751345948129, 'dw': 0.5, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [{'zeta': 1.0, 'dw': 2.0, 'children': [], 'spawn_size': 1}, {'zeta': 1.1, 'dw': 2.1, 'children': [], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 1}], 'spawn_size': 3}], 'spawn_size': 1}]

    assert results_ss_dimension == sample_stack


def test_input():
    zetas = [0.1]
    dws = [0.1, 0.0]

    sample_stack = []
    ss = SpawnStack(sample_stack = sample_stack)

    with pytest.raises(ConfigurationError):
        ss.append_layer(zetas=zetas, dws=dws, stack=ss.sample_stack)
