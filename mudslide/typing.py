# -*- coding: utf-8 -*-
"""Typing prototypes"""

from typing import Any, Union, Iterator, Type
from typing_extensions import Protocol

import numpy as np

ArrayLike = Type[np.ndarray]
DtypeLike = Union[np.float64, np.complex128]

class ElectronicT(Protocol):
    hamiltonian: ArrayLike
    force: ArrayLike
    derivative_coupling: ArrayLike

class DiabaticModelT(Protocol):
    ndim_: int
    nstates_: int

    def __init__(self, representation: str, reference: Any, *args: Any, **kwargs: Any):
        pass

    def V(self, x: ArrayLike) -> ArrayLike:
        pass

    def dV(self, x: ArrayLike) -> ArrayLike:
        pass

class ModelT(Protocol):
    def __init__(self, representation: str, reference: Any, *args: Any, **kwargs: Any):
        pass

    def nstates(self) -> int:
        pass

    def ndim(self) -> int:
        pass

    def compute(self, x: ArrayLike, couplings: Any = None, gradients: Any = None, reference: Any = None) -> None:
        pass

    def update(self, x: ArrayLike, couplings: Any = None, gradients: Any = None) -> 'ModelT':
        pass

class TrajGenT(Protocol):
    def __call__(self, nsamples: int) -> Iterator:
        pass
