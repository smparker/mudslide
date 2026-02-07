#!/usr/bin/env python
"""Parsers for data common across multiple Turbomole modules.

Includes parsers for basis set information, DFT settings, ground state
properties, cartesian gradients, and nonadiabatic coupling vectors.
"""
from __future__ import annotations

import re
from typing import Any, Callable

from .section_parser import ParseSection, ParserProtocol
from .stack_iterator import StackIterator
from .line_parser import LineParser, SimpleLineParser


def fortran_float(x: str) -> float:
    """Convert a Fortran-formatted float string (using D/d exponent) to Python float."""
    x = x.replace("D", "E")
    x = x.replace("d", "E")
    return float(x)


class CompParser(LineParser):
    """Parse separate components of quantities that go elec, nuc, total"""

    def process(self, m: re.Match[str], out: dict[str, Any]) -> None:
        out["elec"].append(float(m.group(1)))
        out["nuc"].append(float(m.group(2)))
        out["total"].append(float(m.group(3)))


class BasisParser(ParseSection):
    """Parser for basis set information (atoms, primitives, contractions)."""
    name = "basis"

    def __init__(self) -> None:
        super().__init__(r"basis set information", r"total:")
        self.parsers = [
            SimpleLineParser(
                r"([a-z]+)\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)\s+(\S+)\s+(\S+)",
                ["atom", "natom", "nprim", "ncont", "nick", "contraction"],
                types=[str, int, int, int, str, str],
                title="list",
                multi=True),
            SimpleLineParser(r"total:\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)",
                             ["natoms", "nprim", "ncont"],
                             types=[int, int, int]),
        ]

    def clean(self, liter: StackIterator, out: dict[str, Any]) -> None:
        bases = {x["nick"] for x in out["list"]}
        if len(bases) == 1:
            out["nick"] = list(bases)[0]
        else:
            out["nick"] = "mixed"


class DFTParser(ParseSection):
    """Parser for DFT settings (grid size, weight derivatives)."""
    name = "dft"

    def __init__(self) -> None:
        super().__init__(r"^\s*density functional\s*$", r"partition sharpness")
        self.parsers = [
            SimpleLineParser(r"spherical gridsize\s*:\s*(\S+)", ["gridsize"],
                             types=[str]),
            SimpleLineParser(r"iterations will be done with (small) grid",
                             ["mgrid"],
                             types=[str]),
            SimpleLineParser(
                r"Derivatives of quadrature weights will be (included)",
                ["weightderivatives"],
                types=[str]),
        ]

    def clean(self, liter: StackIterator, out: dict[str, Any]) -> None:
        if "mgrid" in out:
            out["gridsize"] = "m" + out["gridsize"]
            del out["mgrid"]


class GroundDipole(ParseSection):
    """Parse ground dipole moment"""
    name = "dipole"

    def __init__(self) -> None:
        super().__init__(r"Electric dipole moment",
                         r" z\s+\S+\s+\S+\s+\S+\s+Norm ")
        self.parsers = [
            CompParser(r" x\s+(\S+)\s+(\S+)\s+(\S+)\s+Norm:"),
            CompParser(r" y\s+(\S+)\s+(\S+)\s+(\S+)"),
            CompParser(r" z\s+(\S+)\s+(\S+)\s+(\S+)\s+Norm"),
        ]

    def prepare(self, out: dict[str, Any]) -> dict[str, Any]:
        out[self.name] = {}
        out[self.name]["elec"] = []
        out[self.name]["nuc"] = []
        out[self.name]["total"] = []

        return out[self.name]


class GroundParser(ParseSection):
    """Parser for ground state properties"""
    name = "ground"

    def __init__(self) -> None:
        super().__init__(r"Ground state", r"\S*=====+")
        self.parsers = [GroundDipole()]


class VarLineParser(LineParser):
    """A line parser that can handle variable length lines"""

    def __init__(self, reg: str, title: str, vars_type: Callable[..., Any] = str) -> None:
        super().__init__(reg)
        self.title = title
        self.vars_type = vars_type

    def process(self, m: re.Match[str], out: dict[str, Any]) -> None:
        if self.title not in out:
            out[self.title] = []
        for group in m.groups():
            if group is not None:
                out[self.title].append(self.vars_type(group))


class CoordParser(ParseSection):
    """Parser for atom-indexed Cartesian vector data (gradients and NAC couplings).

    Parses blocks of data formatted as atom labels followed by dx, dy, dz
    components in Fortran notation. After parsing, the clean() method combines
    the components into a single 'd/dR' list of [dx, dy, dz] per atom.
    """
    name = "regex"

    atom_reg: str = r"^\s*ATOM\s+(\d+\ \D+)" + 4 * r"(?:\s+(\d+\ \D+))?" + r"\s*$"

    def __init__(self, head: str, tail: str) -> None:
        super().__init__(head, tail, multi=True)
        parsers: list[ParserProtocol] = [
            VarLineParser(reg=self.atom_reg, title="atom_list", vars_type=str),
        ]
        for coord in "xyz":
            parsers.append(
                VarLineParser(
                    reg=(rf"^\s*d\w*/d{coord}\s+(-*\d+.\d+D(?:\+|-)\d+)" +
                         4 * r"(?:\s*(-*\d+.\d+D(?:\+|-)\d+))?" + r"\s*$"),
                    title=f"d_d{coord}",
                    vars_type=fortran_float))
        self.parsers = parsers

    def clean(self, liter: StackIterator, out: dict[str, Any]) -> None:
        atom_list = [atom.rstrip() for atom in out["atom_list"]]
        out["atom_list"] = atom_list

        # not always sure the d_dx, d_dy, d_dz exist, but if they do, combine them
        # into (natoms, 3) array with xyz contiguous per atom
        components = [
            x for x in ["dE_dx", "dE_dy", "dE_dz", "d_dx", "d_dy", "d_dz"]
            if x in out
        ]
        out["d/dR"] = [
            list(vals) for vals in zip(*[out[c] for c in components])
        ]
        for component in components:
            del out[component]


class NACParser(CoordParser):
    """
    Parser for NAC coupling.
    """
    name = "coupling"

    coupled_states_reg: str = r"^\s*<\s*(\d+)\s*\|\s*\w+/\w+\s*\|\s*(\d+)>\s*$"

    def __init__(self) -> None:
        # Header may or may not have (state/method) at end
        head = r"\s+cartesian\s+nonadiabatic\s+coupling\s+matrix\s+elements(?:\s+\((\d+)/(\w+)\))?"
        tail = r"maximum component of gradient"
        super().__init__(head, tail)
        self.parsers.insert(
            0,
            SimpleLineParser(self.coupled_states_reg,
                             ["bra_state", "ket_state"],
                             types=[int, int]))


# Constants for the two gradient types
EXCITED_STATE_GRADIENT_HEAD: str = (
    r"(?:cartesian\s+gradients\s+of\s+excited\s+state\s+(?P<index>\d+)|"
    r"cartesian\s+gradient\s+of\s+the\s+energy)\s+\((\w+)/(\w+)\)")
GROUND_STATE_GRADIENT_HEAD: str = r"cartesian\s+gradient\s+of\s+the\s+energy\s+\((\w+)/(\w+)\)"


class GradientDataParser(CoordParser):
    """Parser for cartesian gradient data (ground or excited state)."""
    name = "gradient"

    def __init__(self, head: str) -> None:
        # Tail matches end of gradient section: either "resulting FORCE" (when NAC follows),
        # "maximum component of gradient" (when no NAC), or start of NAC section
        tail = r"(?:resulting FORCE|maximum component of gradient|cartesian\s+nonadiabatic)"
        super().__init__(head, tail)

    def prepare(self, out: dict[str, Any]) -> dict[str, Any]:
        dest = super().prepare(out)
        try:
            assert self.lastsearch is not None
            index = self.lastsearch.group("index")
            if index is not None:
                dest["index"] = int(index)
        except (IndexError, AttributeError):
            pass
        return dest
