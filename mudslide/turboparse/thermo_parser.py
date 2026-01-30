#!/usr/bin/env python

from .section_parser import ParseSection
from .line_parser import SimpleLineParser


class ThermoParser(ParseSection):
    name = "thermo"

    def __init__(self):
        super().__init__(r"T H E R M O", r"thermo\s*:\s*all done")
        self.parsers = [
            SimpleLineParser(r"(\S+)\s*VIB\.\s+\S+\s+\S+\s+\S+\s+\S+", names=['T'], types=[float]),
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
