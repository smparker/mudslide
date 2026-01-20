#!/usr/bin/env python

from __future__ import print_function

import re

from .section_parser import ParseSection
from .line_parser import SimpleLineParser
from .common_parser import BasisParser, DFTParser, GradientDataParser, GROUND_STATE_GRADIENT_HEAD


class SCFParser(ParseSection):
    energy = SimpleLineParser(r"\|\s*total energy\s*=\s*(\S+)\s*\|", names=['energy'], types=[float])
    energyOC = SimpleLineParser(r"Total energy \+ OC corr\. =\s*(\S+)", names=['energy+OC'], types=[float])
    basis = BasisParser()
    dftparser = DFTParser()

    parsers = [energy, energyOC, basis]


class RIDFTParser(SCFParser):
    name = 'ridft'

    def __init__(self):
        SCFParser.__init__(self, r"^\s*r i d f t", r"ridft\s*:\s*all done")


class DSCFParser(SCFParser):
    name = 'dscf'

    def __init__(self):
        SCFParser.__init__(self, r"^\s*d s c f", r"dscf\s*:\s*all done")


class GradientModuleParser(ParseSection):
    """Base parser for gradient module output (grad, rdgrad)."""
    dft = DFTParser()
    gradient = GradientDataParser(GROUND_STATE_GRADIENT_HEAD)
    parsers = [dft, gradient]


class RdgradModuleParser(GradientModuleParser):
    """Parser for rdgrad module output."""
    name = 'rdgrad'

    def __init__(self):
        GradientModuleParser.__init__(self, r"^\s*r d g r a d", r"rdgrad\s+:\s*all done")


class GradModuleParser(GradientModuleParser):
    """Parser for grad module output."""
    name = 'grad'

    def __init__(self):
        GradientModuleParser.__init__(self, r"^\s*g r a d", r"grad\s+:\s*all done")
