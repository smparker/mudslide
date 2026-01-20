#!/usr/bin/env python

from __future__ import print_function

import re

from .stack_iterator import StackIterator
from .section_parser import ParseSection
from .scf_parser import RIDFTParser, DSCFParser, RDGRADParser, GRADParser
from .response_parser import EgradParser, EscfParser
from .freeh_parser import FreeHParser
from .thermo_parser import ThermoParser


class TurboParser(ParseSection):
    name = ""

    parsers = [
        RIDFTParser(),
        DSCFParser(),
        RDGRADParser(),
        GRADParser(),
        EgradParser(),
        EscfParser(),
        FreeHParser(),
        ThermoParser()
    ]

    def __init__(self):
        ParseSection.__init__(self, r".*", r"a^")


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

            if (not adv):
                next(liter)
    except StopIteration:
        return out
