#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Unit testing for mudslide"""

import unittest
import sys
import re

import mudslide
import numpy as np

class TestQuadratures(unittest.TestCase):
    """Test Suite for integration quadratures"""
    def setUp(self):
        """Setup function"""
        self.a = 0
        self.b = 1
        self.n = 11

        self.evenx = [ i/(self.n-1) for i in range(self.n) ]
        self.flatw = [0.09090909, 0.09090909, 0.09090909, 0.09090909, 0.09090909, 0.09090909, 0.09090909, 0.09090909, 0.09090909, 0.09090909, 0.09090909]

    def test_midpoint_quadrature(self):
        """Test midpoint quadrature"""
        refx = [0.04545455, 0.13636364, 0.22727273, 0.31818182, 0.40909091, 0.5, 0.59090909, 0.68181818, 0.77272727, 0.86363636, 0.95454545]
        refw = self.flatw
        x, w = mudslide.integration.quadrature(self.n, self.a, self.b, "midpoint")
        self.assertEqual(len(x), self.n)
        self.assertTrue(np.all(np.isclose(x, refx)))

        self.assertEqual(len(w), self.n)
        self.assertTrue(np.all(np.isclose(w, refw)))

    def test_trapezoid_quadrature(self):
        """Test trapezoid quadrature"""
        refx = self.evenx
        refw = [ 0.1 for i in range(self.n) ]
        refw[0] *= 0.5
        refw[-1] *= 0.5

        x, w = mudslide.integration.quadrature(self.n, self.a, self.b, "trapezoid")
        self.assertEqual(len(x), self.n)
        self.assertTrue(np.all(np.isclose(x, refx)))

        self.assertEqual(len(w), self.n)
        self.assertTrue(np.all(np.isclose(w, refw)))

    def test_simpson_quadrature(self):
        """Test Simpson quadrature"""
        refx = self.evenx
        refw = [0.03333333, 0.13333333, 0.06666667, 0.13333333, 0.06666667, 0.13333333,
                 0.06666667, 0.13333333, 0.06666667, 0.13333333, 0.03333333]

        x, w = mudslide.integration.quadrature(self.n, self.a, self.b, "simpson")
        self.assertEqual(len(x), self.n)
        self.assertTrue(np.all(np.isclose(x, refx)))

        self.assertEqual(len(w), self.n)
        self.assertTrue(np.all(np.isclose(w, refw)))

    def test_clenshawcurtis_quadrature(self):
        """Test Clenshaw-Curtis quadrature"""
        refx = [0.        , 0.02447174, 0.0954915 , 0.20610737, 0.3454915 , 0.5       ,
                 0.6545085 , 0.79389263, 0.9045085 , 0.97552826, 1.        ]
        refw = [0.00505051, 0.04728953, 0.09281761, 0.12679417, 0.14960664, 0.15688312,
                 0.14960664, 0.12679417, 0.09281761, 0.04728953, 0.00505051]

        x, w = mudslide.integration.quadrature(self.n, self.a, self.b, "cc")
        self.assertEqual(len(x), self.n)
        self.assertTrue(np.all(np.isclose(x, refx)))

        self.assertEqual(len(w), self.n)
        self.assertTrue(np.all(np.isclose(w, refw)))

    def test_gasslegendre_quadrature(self):
        """Test Gauss-Legendre quadrature"""
        refx = [0.01088567, 0.0564687 , 0.134924  , 0.24045194, 0.36522842, 0.5       ,
                 0.63477158, 0.75954806, 0.865076  , 0.9435313 , 0.98911433]
        refw = [0.02783428, 0.06279018, 0.09314511, 0.11659688, 0.13140227, 0.13646254,
                 0.13140227, 0.11659688, 0.09314511, 0.06279018, 0.02783428]

        x, w = mudslide.integration.quadrature(self.n, self.a, self.b, "gl")
        self.assertEqual(len(x), self.n)
        self.assertTrue(np.all(np.isclose(x, refx)))

        self.assertEqual(len(w), self.n)
        self.assertTrue(np.all(np.isclose(w, refw)))

if __name__ == '__main__':
    unittest.main()
