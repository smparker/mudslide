# -*- coding: utf-8 -*-
"""Helper module for printing model surface"""

from __future__ import print_function

import sys

import argparse
import numpy as np

from .models import scattering_models as models

from typing import Any, List
from .typing import ArrayLike


def main(argv=None, file=sys.stdout) -> None:
    parser = argparse.ArgumentParser(description="Generate potential energy surface scans of two-state models")
    parser.add_argument('-m', '--model', default='simple', choices=[m for m in models], help="Tully model to plot")
    parser.add_argument('-r',
                        '--range',
                        default=(-10.0, 10.0),
                        nargs=2,
                        type=float,
                        help="range over which to plot PES (default: %(default)s)")
    parser.add_argument('-n', default=100, type=int, help="number of points to plot")
    parser.add_argument('-s',
                        '--scan_dimension',
                        default=0,
                        type=int,
                        help="which dimension to scan along for multi-dimensional models")
    parser.add_argument('--x0',
                        nargs='+',
                        default=[0.0],
                        type=float,
                        help="reference point for multi-dimensional models")
    args = parser.parse_args(argv)

    if args.model in models:
        model = models[args.model]()
    else:
        raise Exception("Unknown model chosen")  # the argument parser should prevent this throw from being possible

    start, end = args.range
    samples = args.n

    xr = np.linspace(start, end, samples, dtype=np.float64)

    nstates = model.nstates()
    ndim = model.ndim()

    if len(args.x0) != ndim:
        print("Must provide reference vector of same length as the model problem")
        raise Exception("Expected reference vector of length {}, but received {}".format(ndim, len(args.x0)))

    xx = np.array(args.x0)
    xx[args.scan_dimension] = start

    last_elec = None
    elec = model.update(xx)

    def headprinter() -> str:
        xn = ["x{:d}".format(i) for i in range(ndim)]
        diabats = ["V_%1d" % i for i in range(nstates)]
        energies = ["E_%1d" % i for i in range(nstates)]
        dc = ["d_%1d%1d" % (j, i) for i in range(nstates) for j in range(i)]
        if model == "vibronic":
            forces = ["dE_%1d" % i for i in range(ndim * 2)]
        else:
            forces = ["dE_%1d" % i for i in range(nstates)]

        plist = xn + diabats + energies + dc + forces
        return "#" + " ".join(["%16s" % x for x in plist])

    def lineprinter(x: ArrayLike, model: Any, estates: Any) -> str:
        V = model.V(x)
        ndim = estates.ndim()
        diabats = [V[i, i] for i in range(nstates)]  # type: List[float]
        energies = [estates.hamiltonian()[i, i] for i in range(nstates)]  # type: List[float]
        dc = [estates.derivative_coupling[j, i, 0] for i in range(nstates) for j in range(i)]  # type: List[float]
        forces = [float(-estates.force[i, j]) for i in range(nstates) for j in range(ndim)]  # type: List[float]
        plist = list(x.flatten()) + diabats + energies + dc + forces  # type: List[float]

        return " ".join(["{:16.10f}".format(x) for x in plist])

    #print("# scanning using model {}".format(args.model), file=file)
    #print("# reference point: {}".format(xx), file=file)
    #print("# scan dimension: {}".format(args.scan_dimension), file=file)
    print(headprinter(), file=file)

    for x in xr:
        i = args.scan_dimension
        xx[i] = x
        elec = elec.update(xx, last_elec)
        print(lineprinter(xx, model, elec), file=file)

        last_elec = elec


if __name__ == "__main__":
    main()
