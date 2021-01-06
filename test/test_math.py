#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Unit testing for mudslide"""

import unittest
import sys

from mudslide import poisson_prob_scale
import numpy as np

class TestMath(unittest.TestCase):
    """Test Suite for math functions"""
    def test_poisson_scale(self):
        """Poisson scaling function"""
        args = np.array([0.0, 0.5, 1.0])
        fx = poisson_prob_scale(args)
        refs = np.array([1.0, 0.7869386805747332, 0.6321205588285577])
        for i, x in enumerate(args):
            self.assertAlmostEqual(fx[i], refs[i], places=10)
            self.assertAlmostEqual(poisson_prob_scale(args[i]), refs[i], places=10)

if __name__ == '__main__':
    unittest.main()
