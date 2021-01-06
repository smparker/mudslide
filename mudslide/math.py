# -*- coding: utf-8 -*-
"""Math helper functions"""

from __future__ import division

import numpy as np
import warnings

def poisson_prob_scale(x):
    """Computes (1 - exp(-x))/x which is needed when scaling Poisson probabilities"""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        out = np.where(np.absolute(x) < 1e-3,
                1 - x/2 + x**2/6 - x**3/24,
                -np.expm1(-x)/x)
    return out
