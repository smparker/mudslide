#!/usr/bin/env python

from __future__ import print_function

import re
import json

from .section_parser import ParseSection
from .line_parser import SimpleLineParser


class ThermoParser(ParseSection):
    name = "thermo"

    temp = SimpleLineParser(r"(\S+)\s*VIB\.\s+\S+\s+\S+\s+\S+\s+\S+", names=['T'], types=[float])
    H = SimpleLineParser(r"H\(T\)\s+(\S+)\s+(\S+)\s+(\S+)",
                         names=["H(au)", "H(kcal/mol)", "H(kJ/mol)"],
                         types=[float, float, float])
    S = SimpleLineParser(r"T\*S\s+(\S+)\s+(\S+)\s+(\S+)",
                         names=["ST(au)", "ST(kcal/mol)", "ST(kJ/mol)"],
                         types=[float, float, float])
    G = SimpleLineParser(r"G\(T\)\s+(\S+)\s+(\S+)\s+(\S+)",
                         names=["G(au)", "G(kcal/mol)", "G(kJ/mol)"],
                         types=[float, float, float])

    parsers = [temp, H, S, G]

    def __init__(self):
        ParseSection.__init__(self, r"T H E R M O", r"thermo\s*:\s*all done")
