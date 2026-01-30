#!/usr/bin/env python

from .stack_iterator import StackIterator
from .section_parser import ParseSection
from .scf_parser import RIDFTParser, DSCFParser, RdgradModuleParser, GradModuleParser
from .response_parser import EgradParser, EscfParser
from .freeh_parser import FreeHParser
from .thermo_parser import ThermoParser


class TurboParser(ParseSection):
    name = ""

    def __init__(self):
        super().__init__(r".*", r"a^")
        self.parsers = [
            RIDFTParser(),
            DSCFParser(),
            RdgradModuleParser(),
            GradModuleParser(),
            EgradParser(),
            EscfParser(),
            FreeHParser(),
            ThermoParser()
        ]


def parse_turbo(iterable):
    liter = StackIterator(iterable)
    parser = TurboParser()

    out = {}
    try:
        next(liter)
        while True:
            gotparsed = False
            fnd, adv = parser.parse(liter, out)
            gotparsed = gotparsed or fnd

            if not adv:
                next(liter)
    except StopIteration:
        return out
