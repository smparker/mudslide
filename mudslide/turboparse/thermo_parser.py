#!/usr/bin/env python
"""Parser for Turbomole thermo module output.

Extracts thermochemical quantities: enthalpy H(T), entropy contribution
T*S, and Gibbs free energy G(T) in atomic units, kcal/mol, and kJ/mol.
"""
from __future__ import annotations

from .section_parser import ParseSection
from .line_parser import SimpleLineParser


class ThermoParser(ParseSection):
    """Parser for the thermo module output (H, T*S, G at given temperature)."""
    name = "thermo"

    def __init__(self) -> None:
        super().__init__(r"T H E R M O", r"thermo\s*:\s*all done")
        self.parsers = [
            SimpleLineParser(r"(\S+)\s*VIB\.\s+\S+\s+\S+\s+\S+\s+\S+",
                             names=['T'],
                             types=[float]),
            SimpleLineParser(r"H\(T\)\s+(\S+)\s+(\S+)\s+(\S+)",
                             names=["H(au)", "H(kcal/mol)", "H(kJ/mol)"],
                             types=[float, float, float]),
            SimpleLineParser(r"T\*S\s+(\S+)\s+(\S+)\s+(\S+)",
                             names=["ST(au)", "ST(kcal/mol)", "ST(kJ/mol)"],
                             types=[float, float, float]),
            SimpleLineParser(r"G\(T\)\s+(\S+)\s+(\S+)\s+(\S+)",
                             names=["G(au)", "G(kcal/mol)", "G(kJ/mol)"],
                             types=[float, float, float]),
        ]
