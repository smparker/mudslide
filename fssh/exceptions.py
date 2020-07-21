#!/usr/bin/env python
## @package fssh
#  Module responsible for propagating surface hopping trajectories

from __future__ import print_function, division

from .version import __version__

import copy as cp
import numpy as np

## Exception class indicating that a simulation was terminated while still inside the "interaction region"
class StillInteracting(Exception):
    def __init__(self) -> None:
        Exception.__init__(self, "A simulation ended while still inside the interaction region.")

