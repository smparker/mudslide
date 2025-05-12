#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit testing for boltzmann function"""

import numpy as np
import unittest
from mudslide.math import boltzmann_velocities


class TrajectoryTest(unittest.TestCase):

    def test_boltzmann(self):
        """test for boltzmann_velocity function"""
        # yapf: disable
        mass = np.array ([21874.6618344, 21874.6618344, 21874.6618344,
                          21874.6618344, 21874.6618344, 21874.6618344,
                          1837.15264736,  1837.15264736, 1837.15264736,
                          1837.15264736,  1837.15264736, 1837.15264736,
                          1837.15264736,  1837.15264736, 1837.15264736,
                          1837.15264736,  1837.15264736, 1837.15264736])
        # yapf: enable
        velocities = boltzmann_velocities(mass, temperature=300, remove_translation=False,
                                          scale=True, seed=109)

        # yapf: disable
        results = [ 2.56748743e-04,  3.44471310e-04,  2.46424017e-07,
                   -3.89747528e-04,  2.31250717e-04,  1.20055248e-04,
                   -2.49906132e-04, -5.58142454e-05, -2.39676921e-04,
                    3.22950419e-04,  2.51235241e-04, -9.35539925e-04,
                    7.00590108e-05, -6.98837331e-04, -1.31891980e-03,
                   -8.41937362e-05,  9.68859510e-04,  3.86297562e-04]
        # yapf: enable

        np.testing.assert_almost_equal(results, velocities, decimal=8)

        # yapf: enable
        velocities = boltzmann_velocities(mass, temperature=300, remove_translation=True,
                                          scale=True, seed=109)

        # yapf: disable
        results = [ 3.11566567e-04,  8.12744703e-05,  2.45304540e-05,
                   -3.34929704e-04, -3.19461226e-05,  1.44339278e-04,
                   -1.95088307e-04, -3.19011085e-04, -2.15392891e-04,
                    3.77768244e-04, -1.19615991e-05, -9.11255895e-04,
                    1.24876835e-04, -9.62034171e-04, -1.29463577e-03,
                   -2.93759117e-05, 7.05662670e-04,  4.10581592e-04]
        # yapf: enable

        np.testing.assert_almost_equal(results, velocities, decimal=8)


if __name__ == '__main__':
    unittest.main()
