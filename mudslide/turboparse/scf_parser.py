#!/usr/bin/env python
"""Parsers for Turbomole SCF and gradient module output.

Handles output from ridft, dscf (SCF calculations) and rdgrad, grad
(gradient calculations), extracting energies, convergence status,
basis set info, and gradient data.
"""
from __future__ import annotations

from .section_parser import ParseSection
from .line_parser import BooleanLineParser, SimpleLineParser
from .common_parser import BasisParser, DFTParser, GradientDataParser, GROUND_STATE_GRADIENT_HEAD


class SCFParser(ParseSection):
    """Base parser for SCF module output (energy, convergence, basis, DFT settings)."""

    def __init__(self, head: str, tail: str) -> None:
        super().__init__(head, tail)
        self.parsers = [
            SimpleLineParser(r"\|\s*total energy\s*=\s*(\S+)\s*\|",
                             names=['energy'],
                             types=[float]),
            SimpleLineParser(r"Total energy \+ OC corr\. =\s*(\S+)",
                             names=['energy+OC'],
                             types=[float]),
            BooleanLineParser(
                r"convergence criteria satisfied after\s+\d+\s+iterations",
                r"convergence criteria cannot be satisfied within\s+\d+\s+iterations\s*!",
                "converged",
            ),
            BasisParser(),
            DFTParser(),
        ]


class RIDFTParser(SCFParser):
    """Parser for the ridft (RI-DFT SCF) module output."""
    name = 'ridft'

    def __init__(self) -> None:
        super().__init__(r"^\s*r i d f t", r"ridft\s*:\s*all done")


class DSCFParser(SCFParser):
    """Parser for the dscf (conventional SCF) module output."""
    name = 'dscf'

    def __init__(self) -> None:
        super().__init__(r"^\s*d s c f", r"dscf\s*:\s*all done")


class GradientModuleParser(ParseSection):
    """Base parser for gradient module output (grad, rdgrad)."""

    def __init__(self, head: str, tail: str) -> None:
        super().__init__(head, tail)
        self.parsers = [
            DFTParser(),
            GradientDataParser(GROUND_STATE_GRADIENT_HEAD),
        ]


class RdgradModuleParser(GradientModuleParser):
    """Parser for rdgrad module output."""
    name = 'rdgrad'

    def __init__(self) -> None:
        super().__init__(r"^\s*r d g r a d", r"rdgrad\s+:\s*all done")


class GradModuleParser(GradientModuleParser):
    """Parser for grad module output."""
    name = 'grad'

    def __init__(self) -> None:
        super().__init__(r"^\s*g r a d", r"grad\s+:\s*all done")
