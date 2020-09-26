# -*- coding: utf-8 -*-
"""Helper module for printing model surface"""

from __future__ import print_function

import sys

import argparse
import numpy as np

from .models import models

from typing import Any, List
from .typing import ArrayLike

def main(argv=None, file=sys.stdout) -> None:
    parser = argparse.ArgumentParser(description="Generate potential energy surface scans of two-state models")
    parser.add_argument('-m', '--model', default='simple', choices=[m for m in models], help="Tully model to plot")
    parser.add_argument('-r', '--range', default=(-10.0,10.0), nargs=2, type=float, help="range over which to plot PES (default: %(default)s)")
    parser.add_argument('-n', default=100, type=int, help="number of points to plot")
    args = parser.parse_args(argv)

    if args.model in models:
        model = models[args.model]()
    else:
        raise Exception("Unknown model chosen") # the argument parser should prevent this throw from being possible

    start, end = args.range
    samples = args.n

    xr = np.linspace(start, end, samples, dtype=np.float64)

    nstates = model.nstates()
    last_elec = None
    elec = model.update(start)

    def headprinter() -> str:
        diabats = [ "V_%1d" % i for i in range(nstates) ]
        energies = [ "E_%1d" % i for i in range(nstates) ]
        dc = [ "d_%1d%1d" % (j, i) for i in range(nstates) for j in range(i) ]
        forces = [ "dE_%1d" % i for i in range(nstates) ]

        plist = [ "x" ] + diabats + energies + dc + forces
        return "#" + " ".join([ "%16s" % x for x in plist ])

    def lineprinter(x: ArrayLike, model: Any, estates: Any) -> str:
        V = model.V(x)
        ndim = estates.ndim()
        diabats = [ V[i,i] for i in range(nstates) ] # type: List[float]
        energies = [ estates.hamiltonian[i,i] for i in range(nstates) ] # type: List[float]
        dc = [ estates.derivative_coupling[j,i,0] for i in range(nstates) for j in range(i) ] # type: List[float]
        forces = [ float(-estates.force[i,j]) for i in range(nstates) for j in range(ndim) ] # type: List[float]
        plist = list(x.flatten()) + diabats + energies + dc + forces # type: List[float]

        return " ".join([ "{:16.10f}".format(x) for x in plist ])

    print(headprinter(), file=file)

    for x in xr:
        elec = elec.update(x, last_elec)
        print(lineprinter(x, model, elec), file=file)

        last_elec = elec

if __name__ == "__main__":
    main()
