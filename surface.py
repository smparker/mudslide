#!/usr/bin/env python

from __future__ import print_function

import sys
import argparse
import numpy as np
import tullymodels as tm
import fssh

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate potential energy surface scans of two-state models")
    parser.add_argument('-m', '--model', default='simple', choices=[m for m in tm.modeldict], help="Tully model to plot")
    parser.add_argument('-r', '--range', default=(-10.0,10.0), nargs=2, type=float, help="range over which to plot PES (default: %(default)s)")
    parser.add_argument('-n', default=100, type=int, help="number of points to plot")
    args = parser.parse_args()

    if args.model in tm.modeldict:
        model = tm.modeldict[args.model]()
    else:
        raise Exception("Unknown model chosen") # the argument parser should prevent this throw from being possible

    start, end = args.range
    samples = args.n

    xr = np.linspace(start, end, samples)

    nstates = model.nstates()
    last_elec = None

    def headprinter():
        out = "#%12s" % ("x")
        out += " ".join([ "%12s" % ( "E_%1d" % i) for i in range(nstates)])
        for i in range(nstates):
            out += " ".join([ "%12s" % ( "d_%1d%1d" % (j, i) ) for j in range(i) ])
        out += " ".join([ "%12s" % ( "dE_%1d" % i) for i in range(nstates)])
        return out

    def lineprinter(x, estates):
        ham = estates.hamiltonian
        energies = [ ham[i,i] for i in range(nstates) ]
        out = "%12.6f" % (x)
        out += " ".join([ "%12.6f" % e for e in energies ])
        for i in range(nstates):
            out += " ".join([ "%12.6f" % ( estates.derivative_coupling[j,i] ) for j in range(i) ])
        out += " ".join([ "%12.6f" % tuple(-estates.force[i,:]) for i in range(nstates)])
        return out

    print(headprinter())
    for x in xr:
        elec = fssh.ElectronicStates(model.V(x), model.dV(x), last_elec)
        print(lineprinter(x, elec))

        last_elec = elec
