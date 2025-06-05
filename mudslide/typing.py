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
    """Protocol defining the interface for diabatic models.
    
    This protocol defines the required attributes and methods that a class
    must implement to be considered a valid diabatic model.
    
    Attributes
    ----------
    ndim_ : int
        Number of classical degrees of freedom
    nstates_ : int
        Number of electronic states
    """

    ndim_: int
    nstates_: int

    def __init__(self, representation: str, reference: Any, *args: Any, **kwargs: Any):
        """Initialize the diabatic model.
        
        Parameters
        ----------
        representation : str
            The representation to use ("adiabatic" or "diabatic")
        reference : Any
            The reference electronic state
        *args : Any
            Additional positional arguments
        **kwargs : Any
            Additional keyword arguments
        """
        pass

    def V(self, x: ArrayLike) -> ArrayLike:
        """Compute the diabatic potential energy matrix.
        
        Parameters
        ----------
        x : ArrayLike
            Position at which to compute the potential
            
        Returns
        -------
        ArrayLike
            Potential energy matrix of shape (nstates, nstates)
        """
        pass

    def dV(self, x: ArrayLike) -> ArrayLike:
        """Compute the gradient of the diabatic potential energy matrix.
        
        Parameters
        ----------
        x : ArrayLike
            Position at which to compute the gradient
            
        Returns
        -------
        ArrayLike
            Gradient of potential energy matrix of shape (nstates, nstates, ndim)
        """
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
