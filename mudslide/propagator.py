# -*- coding: utf-8 -*-
"""Propagators for ODEs from quantum dynamics"""

from __future__ import division

import numpy as np

from typing import Any

class Propagator_:
    """Base class for propagators"""
    def __init__(self):
        pass

    def __call__(self, traj: Any, nsteps: int) -> None:
        raise NotImplementedError
