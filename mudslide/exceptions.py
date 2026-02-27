# -*- coding: utf-8 -*-
"""Mudslide exception hierarchy.

All mudslide-specific exceptions inherit from MudslideError, making it
possible to catch any library error with ``except MudslideError``.
"""


class MudslideError(Exception):
    """Base class for all mudslide errors."""


class ConfigurationError(MudslideError):
    """Bad user input, invalid options, dimension mismatches, or unsupported features."""


class ExternalCodeError(MudslideError):
    """An external tool (e.g. Turbomole) crashed or is unavailable."""


class ConvergenceError(ExternalCodeError):
    """SCF or iterative solver convergence failure (recoverable subset of ExternalCodeError)."""


class ComputeError(MudslideError):
    """Runtime numerical or algorithmic error during simulation."""


class MissingDataError(MudslideError):
    """Requested data (forces, couplings) was not computed."""


class MissingForceError(MissingDataError):
    """Requested force/gradient data was not computed."""


class MissingCouplingError(MissingDataError):
    """Requested derivative coupling data was not computed."""
