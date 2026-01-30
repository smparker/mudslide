#!/usr/bin/env python

from .section_parser import ParseSection
from .line_parser import SimpleLineParser
from .common_parser import BasisParser, DFTParser, GradientDataParser, GROUND_STATE_GRADIENT_HEAD


class SCFParser(ParseSection):

    def __init__(self, head, tail):
        super().__init__(head, tail)
        self.parsers = [
            SimpleLineParser(r"\|\s*total energy\s*=\s*(\S+)\s*\|", names=['energy'], types=[float]),
            SimpleLineParser(r"Total energy \+ OC corr\. =\s*(\S+)", names=['energy+OC'], types=[float]),
            BasisParser(),
            DFTParser(),
        ]


class RIDFTParser(SCFParser):
    name = 'ridft'

    def __init__(self):
        super().__init__(r"^\s*r i d f t", r"ridft\s*:\s*all done")


class DSCFParser(SCFParser):
    name = 'dscf'

    def __init__(self):
        super().__init__(r"^\s*d s c f", r"dscf\s*:\s*all done")


class GradientModuleParser(ParseSection):
    """Base parser for gradient module output (grad, rdgrad)."""

    def __init__(self, head, tail):
        super().__init__(head, tail)
        self.parsers = [
            DFTParser(),
            GradientDataParser(GROUND_STATE_GRADIENT_HEAD),
        ]


class RdgradModuleParser(GradientModuleParser):
    """Parser for rdgrad module output."""
    name = 'rdgrad'

    def __init__(self):
        super().__init__(r"^\s*r d g r a d", r"rdgrad\s+:\s*all done")


class GradModuleParser(GradientModuleParser):
    """Parser for grad module output."""
    name = 'grad'

    def __init__(self):
        super().__init__(r"^\s*g r a d", r"grad\s+:\s*all done")
