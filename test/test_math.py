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
        """Test poisson scaling function"""
        self.assertAlmostEqual(poisson_prob_scale(0.0), 1.0, places=10)
        self.assertAlmostEqual(poisson_prob_scale(0.5), 0.7869386805747332, places=10)
        self.assertAlmostEqual(poisson_prob_scale(1.0), 0.6321205588285577, places=10)

if __name__ == '__main__':
    unittest.main()
