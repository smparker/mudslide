#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit testing for boltzmann function"""

import numpy as np
from mudslide.math import boltzmann_velocities


def test_boltzmann():
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
    results = np.array(
            [ 3.52267636e-04,  9.18916486e-05,  2.77349560e-05,
             -3.78682783e-04, -3.61193603e-05,  1.63194840e-04,
             -2.20573399e-04, -3.60684658e-04, -2.43530444e-04,
              4.27117477e-04, -1.35241861e-05, -1.03029655e-03,
              1.41189948e-04, -1.08770818e-03, -1.46375873e-03,
             -3.32133934e-05, 7.97845939e-04,  4.64217351e-04])
    # yapf: enable

    np.testing.assert_almost_equal(results, velocities, decimal=8)
