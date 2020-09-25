# -*- coding: utf-8 -*-
"""Math helper functions"""

from __future__ import division

import numpy as np

def poisson_prob_scale(x):
    """Computes (1 - exp(-x))/x which is needed when scaling Poisson probabilities"""
    if abs(x) < 1e-3:
        return 1 - x/2 + x**2/6 - x**3/24
    else:
        return -np.expm1(-x)/x
