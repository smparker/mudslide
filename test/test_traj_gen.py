#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit testing for mudslide"""

import mudslide
import numpy as np


def get_initial_from_gen(g):
    l = [x for x in g(1)]
    return l[0]


def _setup_traj_gen():
    """Setup function"""
    n = 4
    rng = np.random.default_rng(7)
    x = np.array([1, 2, 3, 4])
    v = np.array([5, 6, 7, 8])
    i = "ground"
    masses = np.abs(rng.normal(0.0, 1e4, size=n))
    print(masses)
    seed = 9
    seed2 = 11
    return n, x, v, i, masses, seed, seed2


def test_const_gen():
    """Test constant generator"""
    n, x, v, i, masses, seed, seed2 = _setup_traj_gen()
    g = mudslide.TrajGenConst(x, v, i, seed=seed)

    xo, vo, io, o = get_initial_from_gen(g)

    assert np.all(np.isclose(xo, x))
    assert np.all(np.isclose(vo, v))
    assert io == "ground"


def test_normal_gen():
    """Test normal generator"""
    n, x, v, i, masses, seed, seed2 = _setup_traj_gen()
    refx = [1.17096384, 8.7987377, 9.12360539, 1.44846462]
    refv = [4.97020305, 5.94726158, 7.05697264, 7.99439356]

    g = mudslide.TrajGenNormal(x, v, i, 10.0, seed=seed, seed_traj=seed2)

    xo, vo, io, o = get_initial_from_gen(g)

    assert np.all(np.isclose(xo, refx))
    assert np.all(np.isclose(vo, refv))
    assert io == "ground"


def test_boltzmann_gen():
    """Test Boltzmann generator"""
    n, x, v, i, masses, seed, seed2 = _setup_traj_gen()
    refv = [0.00389077, 2.41118654, 2.08038404, -1.56240191]  / masses

    g = mudslide.TrajGenBoltzmann(x, masses, 300, i, seed=seed, velocity_seed=seed2)

    xo, vo, io, o = get_initial_from_gen(g)
    print(vo)
    print(refv)
    assert np.all(np.isclose(xo, x))
    assert np.all(np.isclose(vo, refv))
    assert io == "ground"
