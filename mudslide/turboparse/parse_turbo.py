#!/usr/bin/env python
"""Top-level parser for Turbomole output files.

Provides parse_turbo(), which reads a Turbomole output file (or any
iterable of lines) and returns a nested dict of parsed data organized
by Turbomole module (ridft, dscf, egrad, escf, freeh, etc.).
"""

from .stack_iterator import StackIterator
from .section_parser import ParseSection
from .scf_parser import RIDFTParser, DSCFParser, RdgradModuleParser, GradModuleParser
from .response_parser import EgradParser, EscfParser
from .freeh_parser import FreeHParser
from .thermo_parser import ThermoParser


class TurboParser(ParseSection):
    """Root parser that aggregates all Turbomole module parsers.

    Uses head=r".*" to match any line and tail=r"a^" (unmatchable) so it
    never terminates on its own -- parsing runs until the input is exhausted.
    """
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
    """Parse a Turbomole output file and return structured data.

    Args:
        iterable: An iterable of strings (lines), typically an open file
            handle pointing to a Turbomole output file.

    Returns:
        A nested dict keyed by module name (e.g., 'ridft', 'egrad', 'escf',
        'freeh'). Each value contains the parsed data for that module,
        including energies, excited states, gradients, couplings, and
        thermodynamic data as applicable.
    """
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
