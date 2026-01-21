#!/usr/bin/env python

from __future__ import print_function

import re

from .line_parser import LineParser, SimpleLineParser
from .section_parser import ParseSection
from .common_parser import GroundParser, NACParser, GradientDataParser, EXCITED_STATE_GRADIENT_HEAD


class ExcitedDipoleParser(ParseSection):
    """Parser for excited state dipole moments"""
    xc = SimpleLineParser(r"x\s+(\S+)", ["x"], type=float)
    yc = SimpleLineParser(r"y\s+(\S+)", ["y"], type=float)
    zc = SimpleLineParser(r"z\s+(\S+)", ["z"], type=float)

    parsers = [xc, yc, zc]

    def __init__(self, name, head, tail=r"z\s+(\S+)"):
        self.name = name
        ParseSection.__init__(self, head, tail)

    def prepare(self, out):
        out[self.name] = {"x": 0.0, "y": 0.0, "z": 0.0}
        return out[self.name]


class ExcitedParser(ParseSection):
    """Parser for excited state properties"""
    name = "excited_state"

    index = SimpleLineParser(r"(\d+) (singlet |triplet |)([abte1234567890]*) excitation", ["index", "spin", "irrep"],
                             types=[int, str, str])
    energy = SimpleLineParser(r"Excitation energy:\s+(\S+)", ["energy"], type=float)

    diplen = ExcitedDipoleParser("diplen", r"Electric transition dipole moment \(length rep")

    parsers = [index, energy, diplen]

    def __init__(self):
        ParseSection.__init__(self,
                              r"\d+ (singlet |triplet |)[abte1234567890]+ excitation",
                              r"Electric quadrupole transition moment",
                              multi=True)


class MomentParser(LineParser):

    def __init__(self, reg=r"<\s*(\d+)\|mu\|\s*(\d+)>:\s+(\S+)\s+(\S+)\s+(\S+)"):
        super(self.__class__, self).__init__(reg)

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
    moment = MomentParser()

    parsers = [moment]

    def __init__(self, name, head, tail=r"^\s*$"):
        super(self.__class__, self).__init__(head, tail)
        self.name = name


class StateToStateParser(ParseSection):
    name = "state-to-state"

    braket = SimpleLineParser(r"<\s*(\d+)\s*\|\s*W\s*\|\s*(\d+)\s*>", ["bra", "ket"],
                              types=[int, int],
                              first_only=True)
    dipole = ExcitedDipoleParser("diplen", r"Relaxed electric transition dipole moment \(length rep")

    parsers = [braket, dipole]

    def __init__(self):
        super(self.__class__,
              self).__init__(r"<\s*\d+\s*\|\s*W\s*\|\s*\d+\s*>.*(?:transition|difference) moments",
                             r"(<\s*\d+\s*\|\s*W\s*\|\s*\d+\s*>.*(?:transition|difference) moments)|(S\+T\+V CONTRIBUTIONS TO)",
                             multi=True)


class TPAColParser(ParseSection):
    name = "columns"

    col = SimpleLineParser(r"Column:\s+(\S+)", ["column"], type=int)

    xline = SimpleLineParser(r"xx\s+(\S+)\s+xy\s+(\S+)\s+xz\s+(\S+)", ["xx", "xy", "xz"], type=float)
    yline = SimpleLineParser(r"yx\s+(\S+)\s+yy\s+(\S+)\s+yz\s+(\S+)", ["yx", "yy", "yz"], type=float)
    zline = SimpleLineParser(r"zx\s+(\S+)\s+zy\s+(\S+)\s+zz\s+(\S+)", ["zx", "zy", "zz"], type=float)

    fgh = SimpleLineParser(r"\(dF,dG,dH\):\s+(\S+)\s+(\S+)\s+(\S+)", ["df", "dg", "dh"], type=float)
    strength = SimpleLineParser(r"transition strength \[a\.u\.\]:\s*(\S+)\s*(\S+)\s*(\S+)",
                                ["parallel_strength", "perp_strength", "circular_strength"],
                                type=float)
    cross = SimpleLineParser(r"sigma_0 \[1e-50 cm\^4 s\]:\s*(\S+)\s*(\S+)\s*(\S+)",
                             ["parallel_cross", "perp_cross", "circular_cross"],
                             type=float)

    parsers = [col, xline, yline, zline, fgh, strength, cross]

    def __init__(self):
        ParseSection.__init__(self, r"Column:s+(\S+)", r"sigma_0", multi=True)


class TPAParser(TPAColParser):
    name = "tpa"

    state_symm = SimpleLineParser(r"Two-photon absorption amplitudes for transition to " +
                                  r"the\s+(\d+)\S+\s+electronic excitation in symmetry\s+(\S+)", ["state", "irrep"],
                                  types=[int, str])
    exc_energy = SimpleLineParser(r"Exc\. energy:\s+(\S+)\s+Hartree,\s+(\S+)", ["E (H)", "E (eV)"], type=float)

    omega1 = SimpleLineParser(r"omega_1\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)", ["w1 (H)", "w1 (eV)", "w1 (nm)", "w1 (rcm)"],
                              type=float)
    omega2 = SimpleLineParser(r"omega_2\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)", ["w2 (H)", "w2 (eV)", "w2 (nm)", "w2 (rcm)"],
                              type=float)

    col_section = TPAColParser()

    parsers = [
        state_symm, exc_energy, omega1, omega2, col_section, TPAColParser.xline, TPAColParser.yline,
        TPAColParser.zline, TPAColParser.fgh, TPAColParser.strength, TPAColParser.cross
    ]

    def __init__(self):
        ParseSection.__init__(self,
                              r"Two-photon absorption amplitudes for transition to " +
                              r"the\s+(\d+)\S+\s+electronic excitation in symmetry\s+(\S+)",
                              r"sigma_0",
                              multi=True)


class HyperParser(ParseSection):
    name = "hyper"

    freq_hartree = SimpleLineParser(r"Frequencies:\s+(\S+)\s+(\S+)", ["omega_1", "omega_2"], type=float)
    freq_ev = SimpleLineParser(r"Frequencies / eV:\s+(\S+)\s+(\S+)", ["omega_1 (eV)", "omega_2 (eV)"], type=float)
    freq_nm = SimpleLineParser(r"Frequencies / nm:\s+(\S+)\s+(\S+)", ["omega_1 (nm)", "omega_2 (nm)"], type=str)
    freq_rcm = SimpleLineParser(r"Frequencies / cm\^\(-1\):\s+(\S+)\s+(\S+)", ["omega_1 (rcm)", "omega_2 (rcm)"],
                                type=float)

    elements = [
        SimpleLineParser(r"x%2s\s+(\S+)\s+y%2s\s+(\S+)\s+z%2s\s+(\S+)" % (xy, xy, xy), ["x" + xy, "y" + xy, "z" + xy],
                         type=float) for xy in ["xx", "yx", "zx", "xy", "yy", "zy", "xz", "yz", "zz"]
    ]

    scalar = SimpleLineParser(r"scalar norm:\s+(\S+)", ["scalar"], type=float)
    vectors = SimpleLineParser(r"vector norms:\s+(\S+)\s+(\S+)\s+(\S+)", ["v1", "v2", "v3"], type=float)
    dev1 = SimpleLineParser(r"deviator 1 eigenvalues:\s+(\S+)\s+(\S+)\s+(\S+)\s+", ["d1_1", "d1_2", "d1_3"],
                            type=float)
    dev2 = SimpleLineParser(r"deviator 2 eigenvalues:\s+(\S+)\s+(\S+)\s+(\S+)\s+", ["d2_1", "d2_2", "d2_3"],
                            type=float)
    sep = SimpleLineParser(r"septor nom:\s+(\S+)", ["sep"], type=float)

    parsers = [freq_hartree, freq_ev, freq_nm, freq_rcm, scalar, vectors, dev1, dev2, sep] + elements

    def __init__(self):
        ParseSection.__init__(self, r"(\d+)\S+ pair of frequencies", r"septor norm", multi=True)


class EgradEscfParser(ParseSection):
    ground = GroundParser()
    excited = ExcitedParser()
    rel_moment = ExcitedMoments("relaxed moments", r"Fully relaxed moments of the excited states")
    rel_trans = ExcitedMoments("relaxed transitions", r"Fully relaxed state-to-state transition moments")
    unrel_moment = ExcitedMoments("unrelaxed moments", r"Unrelaxed moments of the excited states")
    unrel_trans = ExcitedMoments("unrelaxed transitions", r"Unrelaxed state-to-state transition moments")
    tpa = TPAParser()
    hyper = HyperParser()
    state_to_state = StateToStateParser()
    coupling = NACParser()
    gradient = GradientDataParser(EXCITED_STATE_GRADIENT_HEAD)

    parsers = [
        ground, excited, rel_moment, rel_trans, unrel_moment, unrel_trans, tpa, hyper, state_to_state, coupling,
        gradient
    ]


class EgradParser(EgradEscfParser):
    name = "egrad"

    def __init__(self):
        ParseSection.__init__(self, r"^\s*e g r a d", r"egrad\s*:\s*all done")


class EscfParser(EgradEscfParser):
    name = "escf"

    def __init__(self):
        ParseSection.__init__(self, r"^\s*e s c f", r"escf\s*:\s*all done")
