#!/usr/bin/env python

from __future__ import print_function

import re

from .section_parser import ParseSection
from .line_parser import LineParser, SimpleLineParser


def fortran_float(x):
    x = x.replace("D", "E")
    x = x.replace("d", "E")
    return float(x)


class CompParser(LineParser):
    """Parse separate components of quantities that go elec, nuc, total"""

    def __init__(self, reg):
        super(self.__class__, self).__init__(reg)

    def process(self, m, out):
        out["elec"].append(float(m.group(1)))
        out["nuc"].append(float(m.group(2)))
        out["total"].append(float(m.group(3)))


class BasisParser(ParseSection):
    name = "basis"

    bas = SimpleLineParser(r"([a-z]+)\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)\s+(\S+)\s+(\S+)",
                           ["atom", "natom", "nprim", "ncont", "nick", "contraction"],
                           types=[str, int, int, int, str, str],
                           title="list",
                           multi=True)

    tot = SimpleLineParser(r"total:\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)", ["natoms", "nprim", "ncont"],
                           types=[int, int, int])

    parsers = [bas, tot]

    def __init__(self):
        super(self.__class__, self).__init__(r"basis set information", r"total:")

    def clean(self, liter, out):
        bases = set([x["nick"] for x in out["list"]])
        if len(bases) == 1:
            out["nick"] = list(bases)[0]
        else:
            out["nick"] = "mixed"


class DFTParser(ParseSection):
    name = "dft"

    gridsize = SimpleLineParser(r"spherical gridsize\s*:\s*(\S+)", ["gridsize"], types=[str])
    mgrid = SimpleLineParser(r"iterations will be done with (small) grid", ["mgrid"], types=[str])
    weightder = SimpleLineParser(r"Derivatives of quadrature weights will be (included)", ["weightderivatives"],
                                 types=[str])

    parsers = [gridsize, mgrid, weightder]

    def __init__(self):
        super(self.__class__, self).__init__(r"^\s*density functional\s*$", r"partition sharpness")

    def clean(self, liter, out):
        if "mgrid" in out:
            out["gridsize"] = "m" + out["gridsize"]
            del out["mgrid"]


class GroundDipole(ParseSection):
    """Parse ground dipole moment"""
    name = "dipole"

    xc = CompParser(r" x\s+(\S+)\s+(\S+)\s+(\S+)\s+Norm:")
    yc = CompParser(r" y\s+(\S+)\s+(\S+)\s+(\S+)")
    zc = CompParser(r" z\s+(\S+)\s+(\S+)\s+(\S+)\s+Norm")

    parsers = [xc, yc, zc]

    def __init__(self):
        super(self.__class__, self).__init__(r"Electric dipole moment", r" z\s+\S+\s+\S+\s+\S+\s+Norm ")

    def prepare(self, out):
        out[self.name] = {}
        out[self.name]["elec"] = []
        out[self.name]["nuc"] = []
        out[self.name]["total"] = []

        return out[self.name]


class GroundParser(ParseSection):
    """Parser for ground state properties"""
    name = "ground"
    dipole = GroundDipole()

    parsers = [dipole]

    def __init__(self):
        super(self.__class__, self).__init__(r"Ground state", r"\S*=====+")


class VarLineParser(LineParser):
    """A line parser that can handle variable length lines"""

    def __init__(self, reg, title, vars_type=str):
        self.reg = reg
        self.title = title
        self.vars_type = vars_type

        super(self.__class__, self).__init__(self.reg)

    def process(self, match, out):
        if self.title not in out.keys():
            out[self.title] = []
        for group in match.groups():
            if group is not None:
                out[self.title].append(self.vars_type(group))


class CoordParser(ParseSection):
    """
    Parser for regex for parsing gradients and NAC coupling.
    """
    name = "regex"

    atom_reg = r"^\s*ATOM\s+(\d+\ \D+)" + 4 * r"(?:\s+(\d+\ \D))?" + r"\s*$"
    atom_list = VarLineParser(reg=atom_reg, title="atom_list", vars_type=str)

    d_dcoord = [
        VarLineParser(reg=rf"^\s*d\w*/d{coord}\s+(-*\d+.\d+D(?:\+|-)\d+)" + 4 * r"(?:\s+(-*\d+.\d+D(?:\+|-)\d+))?" +
                      r"\s*$",
                      title=f"d_d{coord}",
                      vars_type=fortran_float) for coord in "xyz"
    ]

    def __init__(self, head, tail):
        self.parsers = [self.atom_list]
        self.parsers.extend(self.d_dcoord)
        ParseSection.__init__(self, head, tail, multi=True)


class NACParser(CoordParser):
    """
    Parser for NAC coupling.
    """
    name = "coupling"

    coupled_states_reg = r"^\s*<\s*(\d+)\s*\|\s*\w+/\w+\s*\|\s*(\d+)>\s*$"
    coupled_states = SimpleLineParser(coupled_states_reg, ["bra_state", "ket_state"], types=[int, int])

    def __init__(self):
        head = r" cartesian\s+nonadiabatic\s+coupling\s+matrix\s+elements\s+\((\d+)/(\w+)\)"
        tail = r"maximum component of gradient"
        CoordParser.__init__(self, head, tail)
        self.parsers.insert(0, self.coupled_states)

    def clean(self, liter, out):
        atom_list = [atom.rstrip() for atom in out["atom_list"]]
        out["atom_list"] = atom_list
        out["d/dR"] = [[val for val in out[component]] for component in ["d_dx", "d_dy", "d_dz"]]
        del out["d_dx"]
        del out["d_dy"]
        del out["d_dz"]


# Constants for the two gradient types
EXCITED_STATE_GRADIENT_HEAD = r" cartesian\s+gradients\s+of\s+excited\s+state\s+[0-9 ]+\((\w+)/(\w+)\)"
GROUND_STATE_GRADIENT_HEAD = r"cartesian\s+gradient\s+of\s+the\s+energy\s+\((\w+)/(\w+)\)"


class GradientDataParser(CoordParser):
    """Parser for cartesian gradient data (ground or excited state)."""
    name = "gradient"

    def __init__(self, head):
        tail = r"maximum component of gradient"
        CoordParser.__init__(self, head, tail)

    def clean(self, liter, out):
        atom_list = [atom.rstrip() for atom in out["atom_list"]]
        out["atom_list"] = atom_list
