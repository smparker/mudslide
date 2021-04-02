#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Unit testing for mudslide"""

import unittest
import sys
import re

import mudslide
import numpy as np

def get_initial_from_gen(g):
    l = [ x for x in g(1) ]
    return l[0]

class TestTrajGen(unittest.TestCase):
    """Test Suite for trajectory initial state generators"""
    def setUp(self):
        """Setup function"""
        self.n = 4
        rng = np.random.default_rng(7)
        self.x = np.array([1, 2, 3, 4])
        self.p = np.array([5, 6, 7, 8])
        self.i = "ground"
        self.masses = np.abs(rng.normal(0.0, 1e4, size=self.n))
        self.seed = 9
        self.seed2 = 11

    def test_const_gen(self):
        """Test constant generator"""
        g = mudslide.TrajGenConst(self.x, self.p, self.i, seed=self.seed)

        x, p, i, o = get_initial_from_gen(g)

        self.assertTrue(np.all(np.isclose(x, self.x)))
        self.assertTrue(np.all(np.isclose(p, self.p)))
        self.assertEqual(i, "ground")

    def test_normal_gen(self):
        """Test normal generator"""
        refx = [1.17096384, 8.7987377,  9.12360539, 1.44846462]
        refp = [4.97020305, 5.94726158, 7.05697264, 7.99439356]

        g = mudslide.TrajGenNormal(self.x, self.p, self.i, 10.0,
                seed=self.seed, seed_traj=self.seed2)

        x, p, i, o = get_initial_from_gen(g)

        self.assertTrue(np.all(np.isclose(x, refx)))
        self.assertTrue(np.all(np.isclose(p, refp)))
        self.assertEqual(i, "ground")

    def test_boltzmann_gen(self):
        """Test Boltzmann generator"""
        refp = [ 0.00389077, 2.41118654, 2.08038404, -1.56240191]

        g = mudslide.TrajGenBoltzmann(self.x, self.masses, 300, self.i,
                seed=self.seed, momentum_seed=self.seed2)

        x, p, i, o = get_initial_from_gen(g)

        self.assertTrue(np.all(np.isclose(x, self.x)))
        self.assertTrue(np.all(np.isclose(p, refp)))
        self.assertEqual(i, "ground")

if __name__ == '__main__':
    unittest.main()
