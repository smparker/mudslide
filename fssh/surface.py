#!/usr/bin/env python
## @package surface
#  Helper module for printing model surface

from __future__ import print_function

import argparse
import numpy as np
import fssh.models as tm

from typing import Any
from .typing import ArrayLike

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate potential energy surface scans of two-state models")
    parser.add_argument('-m', '--model', default='simple', choices=[m for m in tm.models], help="Tully model to plot")
    parser.add_argument('-r', '--range', default=(-10.0,10.0), nargs=2, type=float, help="range over which to plot PES (default: %(default)s)")
    parser.add_argument('-n', default=100, type=int, help="number of points to plot")
    args = parser.parse_args()

    if args.model in tm.models:
        model = tm.models[args.model]()
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
        diabats = [ V[i,i] for i in range(nstates) ]
        energies = [ estates.hamiltonian[i,i] for i in range(nstates) ]
        dc = [ estates.derivative_coupling[j,i] for i in range(nstates) for j in range(i) ]
        forces = [ -estates.force[i,:] for i in range(nstates) ]
        plist = [ x ] + diabats + energies + dc + forces

        return " ".join([ "%16.10f" % x for x in plist ])

    print(headprinter())

    for x in xr:
        elec = elec.update(x, last_elec)
        print(lineprinter(x, model, elec))

        last_elec = elec

if __name__ == "__main__":
    main()
