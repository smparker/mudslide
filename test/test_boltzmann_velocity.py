#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Unit testing for Turbomole_class"""

import numpy as np
import os
import shutil
import unittest
import sys
from pathlib import Path
import mudslide
import yaml

from mudslide import TMModel
from mudslide.tracer import YAMLTrace
from mudslide.math import boltzmann_velocities

class TrajectoryTest(unittest.TestCase):
    def test_boltzmann(self):
        """test for boltzmann_velocity function"""
        mass = np.array ([ 
                        21874.6618344, 21874.6618344,21874.6618344,
                        21874.6618344,21874.6618344,21874.6618344, 
                        1837.15264736, 1837.15264736, 1837.15264736, 
                        1837.15264736, 1837.15264736 , 1837.15264736 , 
                        1837.15264736 , 1837.15264736 , 1837.15264736 , 
                        1837.15264736 , 1837.15264736 , 1837.15264736 
                        ])
        velocities = boltzmann_velocities(mass, temperature=300, seed = 109)

        results = [ 2.56748743e-04,  3.44471310e-04,  2.46424017e-07, -3.89747528e-04,
                  2.31250717e-04,  1.20055248e-04, -2.49906132e-04, -5.58142454e-05,
                 -2.39676921e-04,  3.22950419e-04,  2.51235241e-04, -9.35539925e-04,
                  7.00590108e-05, -6.98837331e-04, -1.31891980e-03, -8.41937362e-05,
                  9.68859510e-04,  3.86297562e-04]

        np.testing.assert_almost_equal(results, velocities, decimal = 8)

if __name__ == '__main__':
    unittest.main()
