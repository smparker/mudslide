#!/usr/bin/env python
## @package surface
#  Helper module for printing model surface

# fssh: program to run surface hopping simulations for model problems
# Copyright (C) 2018, Shane Parker <smparker@uci.edu>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import print_function

import argparse
import numpy as np
import fssh.models as tm

def main():
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

    xr = np.linspace(start, end, samples)

    nstates = model.nstates()
    last_elec = None
    elec = model.update(start)

    def headprinter():
        diabats = [ "V_%1d" % i for i in range(nstates) ]
        energies = [ "E_%1d" % i for i in range(nstates) ]
        dc = [ "d_%1d%1d" % (j, i) for i in range(nstates) for j in range(i) ]
        forces = [ "dE_%1d" % i for i in range(nstates) ]

        plist = [ "x" ] + diabats + energies + dc + forces
        return "#" + " ".join([ "%16s" % x for x in plist ])

    def lineprinter(x, model, estates):
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
