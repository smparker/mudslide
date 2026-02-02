#!/usr/bin/env python

from .line_parser import LineParser, SimpleLineParser, BooleanLineParser
from .section_parser import ParseSection
from .common_parser import GroundParser, NACParser, GradientDataParser, EXCITED_STATE_GRADIENT_HEAD, fortran_float


class ExcitedDipoleParser(ParseSection):
    """Parser for excited state dipole moments"""

    def __init__(self, name, head, tail=r"z\s+(\S+)"):
        super().__init__(head, tail)
        self.name = name
        self.parsers = [
            SimpleLineParser(r"x\s+(\S+)", ["x"], converter=float),
            SimpleLineParser(r"y\s+(\S+)", ["y"], converter=float),
            SimpleLineParser(r"z\s+(\S+)", ["z"], converter=float),
        ]

    def prepare(self, out):
        out[self.name] = {"x": 0.0, "y": 0.0, "z": 0.0}
        return out[self.name]


class ExcitedParser(ParseSection):
    """Parser for excited state properties"""
    name = "excited_state"

    def __init__(self):
        super().__init__(r"\d+ (singlet |triplet |)[abte1234567890]+ excitation",
                         r"Electric quadrupole transition moment",
                         multi=True)
        self.parsers = [
            SimpleLineParser(r"(\d+) (singlet |triplet |)([abte1234567890]*) excitation",
                             ["index", "spin", "irrep"],
                             types=[int, str, str]),
            SimpleLineParser(r"Excitation energy:\s+(\S+)", ["energy"], converter=float),
            ExcitedDipoleParser("diplen", r"Electric transition dipole moment \(length rep"),
        ]


class MomentParser(LineParser):

    def __init__(self, reg=r"<\s*(\d+)\|mu\|\s*(\d+)>:\s+(\S+)\s+(\S+)\s+(\S+)"):
        super().__init__(reg)

    def process(self, m, out):
        i, j = int(m.group(1)), int(m.group(2))
        dip = [float(m.group(x)) for x in range(3, 6)]

        if i not in out:
            out[i] = {}
        out[i][j] = {"diplen": dip}

        if j not in out:
            out[j] = {}
        out[j][i] = {"diplen": dip}


class ExcitedMoments(ParseSection):

    def __init__(self, name, head, tail=r"^\s*$"):
        super().__init__(head, tail)
        self.name = name
        self.parsers = [MomentParser()]


class StateToStateParser(ParseSection):
    name = "state-to-state"

    def __init__(self):
        super().__init__(r"<\s*\d+\s*\|\s*W\s*\|\s*\d+\s*>.*(?:transition|difference) moments",
                         r"(<\s*\d+\s*\|\s*W\s*\|\s*\d+\s*>.*(?:transition|difference) moments)|(S\+T\+V CONTRIBUTIONS TO)",
                         multi=True)
        self.parsers = [
            SimpleLineParser(r"<\s*(\d+)\s*\|\s*W\s*\|\s*(\d+)\s*>", ["bra", "ket"],
                             types=[int, int],
                             first_only=True),
            ExcitedDipoleParser("diplen", r"Relaxed electric transition dipole moment \(length rep"),
        ]


class TPAColParser(ParseSection):
    name = "columns"

    def __init__(self):
        super().__init__(r"Column:s+(\S+)", r"sigma_0", multi=True)
        self.parsers = self._make_parsers()

    @staticmethod
    def _make_parsers():
        return [
            SimpleLineParser(r"Column:\s+(\S+)", ["column"], converter=int),
            SimpleLineParser(r"xx\s+(\S+)\s+xy\s+(\S+)\s+xz\s+(\S+)", ["xx", "xy", "xz"], converter=float),
            SimpleLineParser(r"yx\s+(\S+)\s+yy\s+(\S+)\s+yz\s+(\S+)", ["yx", "yy", "yz"], converter=float),
            SimpleLineParser(r"zx\s+(\S+)\s+zy\s+(\S+)\s+zz\s+(\S+)", ["zx", "zy", "zz"], converter=float),
            SimpleLineParser(r"\(dF,dG,dH\):\s+(\S+)\s+(\S+)\s+(\S+)", ["df", "dg", "dh"], converter=float),
            SimpleLineParser(r"transition strength \[a\.u\.\]:\s*(\S+)\s*(\S+)\s*(\S+)",
                             ["parallel_strength", "perp_strength", "circular_strength"],
                             converter=float),
            SimpleLineParser(r"sigma_0 \[1e-50 cm\^4 s\]:\s*(\S+)\s*(\S+)\s*(\S+)",
                             ["parallel_cross", "perp_cross", "circular_cross"],
                             converter=float),
        ]


class TPAParser(ParseSection):
    name = "tpa"

    def __init__(self):
        super().__init__(r"Two-photon absorption amplitudes for transition to " +
                         r"the\s+(\d+)\S+\s+electronic excitation in symmetry\s+(\S+)",
                         r"sigma_0",
                         multi=True)
        self.parsers = [
            SimpleLineParser(r"Two-photon absorption amplitudes for transition to " +
                             r"the\s+(\d+)\S+\s+electronic excitation in symmetry\s+(\S+)",
                             ["state", "irrep"],
                             types=[int, str]),
            SimpleLineParser(r"Exc\. energy:\s+(\S+)\s+Hartree,\s+(\S+)", ["E (H)", "E (eV)"],
                             converter=float),
            SimpleLineParser(r"omega_1\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)",
                             ["w1 (H)", "w1 (eV)", "w1 (nm)", "w1 (rcm)"],
                             converter=float),
            SimpleLineParser(r"omega_2\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)",
                             ["w2 (H)", "w2 (eV)", "w2 (nm)", "w2 (rcm)"],
                             converter=float),
            TPAColParser(),
        ] + TPAColParser._make_parsers()


class HyperParser(ParseSection):
    name = "hyper"

    def __init__(self):
        super().__init__(r"(\d+)\S+ pair of frequencies", r"septor norm", multi=True)
        self.parsers = [
            SimpleLineParser(r"Frequencies:\s+(\S+)\s+(\S+)", ["omega_1", "omega_2"], converter=float),
            SimpleLineParser(r"Frequencies / eV:\s+(\S+)\s+(\S+)", ["omega_1 (eV)", "omega_2 (eV)"],
                             converter=float),
            SimpleLineParser(r"Frequencies / nm:\s+(\S+)\s+(\S+)", ["omega_1 (nm)", "omega_2 (nm)"],
                             converter=str),
            SimpleLineParser(r"Frequencies / cm\^\(-1\):\s+(\S+)\s+(\S+)", ["omega_1 (rcm)", "omega_2 (rcm)"],
                             converter=float),
            SimpleLineParser(r"scalar norm:\s+(\S+)", ["scalar"], converter=float),
            SimpleLineParser(r"vector norms:\s+(\S+)\s+(\S+)\s+(\S+)", ["v1", "v2", "v3"], converter=float),
            SimpleLineParser(r"deviator 1 eigenvalues:\s+(\S+)\s+(\S+)\s+(\S+)\s+", ["d1_1", "d1_2", "d1_3"],
                             converter=float),
            SimpleLineParser(r"deviator 2 eigenvalues:\s+(\S+)\s+(\S+)\s+(\S+)\s+", ["d2_1", "d2_2", "d2_3"],
                             converter=float),
            SimpleLineParser(r"septor nom:\s+(\S+)", ["sep"], converter=float),
        ] + [
            SimpleLineParser(rf"x{xy}\s+(\S+)\s+y{xy}\s+(\S+)\s+z{xy}\s+(\S+)",
                             [f"x{xy}", f"y{xy}", f"z{xy}"],
                             converter=float)
            for xy in ["xx", "yx", "zx", "xy", "yy", "zy", "xz", "yz", "zz"]
        ]


class _DavidsonIterationParsers:
    """Common parsers for Block Davidson iteration sections"""

    @staticmethod
    def make_parsers():
        return [
            SimpleLineParser(r"^\s*(\d+)\s+\S+\s+\d+\s+(\S+)\s*$", ["step", "max_residual_norm"],
                             types=[int, fortran_float],
                             title="iterations",
                             multi=True),
            BooleanLineParser(r"^\s*converged!", r"not converged!", "converged"),
        ]


class CPKSParser(ParseSection):
    """Parser for CPKS iterations"""
    name = "cpks"

    def __init__(self):
        super().__init__(r"CPKS right-hand side", r"(not )?converged!")
        self.parsers = _DavidsonIterationParsers.make_parsers()


class DavidsonParser(ParseSection):
    """Parser for excitation vector Davidson iterations"""
    name = "davidson"

    def __init__(self):
        super().__init__(r"^\s*excitation vector\s*$", r"(not )?converged!")
        self.parsers = _DavidsonIterationParsers.make_parsers()


class EgradEscfParser(ParseSection):

    def __init__(self, head, tail):
        super().__init__(head, tail)
        self.parsers = [
            DavidsonParser(),
            CPKSParser(),
            GroundParser(),
            ExcitedParser(),
            ExcitedMoments("relaxed moments", r"Fully relaxed moments of the excited states"),
            ExcitedMoments("relaxed transitions", r"Fully relaxed state-to-state transition moments"),
            ExcitedMoments("unrelaxed moments", r"Unrelaxed moments of the excited states"),
            ExcitedMoments("unrelaxed transitions", r"Unrelaxed state-to-state transition moments"),
            TPAParser(),
            HyperParser(),
            StateToStateParser(),
            NACParser(),
            GradientDataParser(EXCITED_STATE_GRADIENT_HEAD),
        ]


class EgradParser(EgradEscfParser):
    name = "egrad"

    def __init__(self):
        super().__init__(r"^\s*e g r a d", r"egrad\s*:\s*all done")


class EscfParser(EgradEscfParser):
    name = "escf"

    def __init__(self):
        super().__init__(r"^\s*e s c f", r"escf\s*:\s*all done")
