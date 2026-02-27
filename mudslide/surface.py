# -*- coding: utf-8 -*-
"""Helper module for printing model surface"""

from __future__ import annotations

from typing import Any, List, TYPE_CHECKING

import argparse
import sys

import numpy as np
from numpy.typing import ArrayLike

from .exceptions import ConfigurationError
from .models.scattering_models import scattering_models as models
from .version import get_version_info

if TYPE_CHECKING:
    from .models.electronics import DiabaticModel_, ElectronicModel_


def add_surface_parser(subparsers: Any) -> None:
    """ Should accept a subparser object from argparse, add new subcommand, and then add arguments
    """
    parser = subparsers.add_parser(
        'surface',
        help="Generate potential energy surface scans of two-state models")
    add_surface_arguments(parser)

    parser.set_defaults(func=surface_wrapper)


def add_surface_arguments(parser: Any) -> None:
    """ Add arguments to the parser object

    Note, this is spun out so that surface can act as a main function and a subcommand.
    The main function will eventually be deprecated and removed.
    """
    parser.add_argument('-m',
                        '--model',
                        default='simple',
                        choices=[m for m in models],
                        help="Tully model to plot")
    parser.add_argument(
        '-r',
        '--range',
        default=(-10.0, 10.0),
        nargs=2,
        type=float,
        help="range over which to plot PES (default: %(default)s)")
    parser.add_argument('-n',
                        default=100,
                        type=int,
                        help="number of points to plot")
    parser.add_argument(
        '-s',
        '--scan_dimension',
        default=0,
        type=int,
        help="which dimension to scan along for multi-dimensional models")
    parser.add_argument('--x0',
                        nargs='+',
                        default=[0.0],
                        type=float,
                        help="reference point for multi-dimensional models")
    parser.add_argument('-o',
                        '--output',
                        default=sys.stdout,
                        type=argparse.FileType('w'),
                        help="output file")


def surface_wrapper(args: Any) -> None:
    """ Wrapper function for surface

    This function is called by the main function, and is used to wrap the
    surface function so that it can be called by the subcommand interface.
    """
    surface_main(args.model,
                 args.range,
                 args.n,
                 args.scan_dimension,
                 args.x0,
                 output=args.output)


def main(argv: List[str] | None = None) -> None:
    """ Main function for surface

    Deprecated
    """
    parser = argparse.ArgumentParser(
        description=
        "Generate potential energy surface scans of two-state models",
        epilog=get_version_info(),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v',
                        '--version',
                        action='version',
                        version=get_version_info())

    add_surface_arguments(parser)
    args = parser.parse_args(argv)

    surface_wrapper(args)



def surface_main(model_name: str, scan_range: List[float], n: int,
                 scan_dimension: int, x0: List[float],
                 output: Any) -> None:
    """ Main function for surface"""
    if model_name in models:
        model = models[model_name]()
    else:
        raise ConfigurationError(
            "Unknown model chosen"
        )  # the argument parser should prevent this throw from being possible

    start, end = scan_range
    samples = n

    xr = np.linspace(start, end, samples, dtype=np.float64)

    nstates = model.nstates
    ndof = model.ndof

    if len(x0) != ndof:
        print(
            "Must provide reference vector of same length as the model problem")
        raise ConfigurationError(
            f"Expected reference vector of length {ndof}, but received {len(x0)}"
        )

    xx = np.array(x0)
    xx[scan_dimension] = start

    last_elec = None
    elec = model.update(xx)

    def headprinter() -> str:
        xn = [f"x{i}" for i in range(ndof)]
        diabats = [f"V_{i}" for i in range(nstates)]
        energies = [f"E_{i}" for i in range(nstates)]
        dc = [f"d_{j}{i}" for i in range(nstates) for j in range(i)]
        if model_name == "vibronic":
            forces = [f"dE_{i}" for i in range(ndof * 2)]
        else:
            forces = [f"dE_{i}" for i in range(nstates)]

        plist = xn + diabats + energies + dc + forces
        return "#" + " ".join(f"{x:>16s}" for x in plist)

    def lineprinter(x: np.ndarray, model: DiabaticModel_, estates: ElectronicModel_) -> str:
        V = model.V(x)
        ndof = estates.ndof
        diabats = [V[i, i] for i in range(nstates)]  # type: List[float]
        energies = [estates.hamiltonian[i, i] for i in range(nstates)
                   ]  # type: List[float]
        dc = [
            estates._derivative_coupling[j, i, 0]
            for i in range(nstates)
            for j in range(i)
        ]  # type: List[float]
        forces = [
            float(-estates.force(i)[j])
            for i in range(nstates)
            for j in range(ndof)
        ]  # type: List[float]
        plist = list(
            x.flatten()) + diabats + energies + dc + forces  # type: List[float]

        return " ".join(f"{x:16.10f}" for x in plist)

    #print("# scanning using model {}".format(model), file=output)
    #print("# reference point: {}".format(xx), file=output)
    #print("# scan dimension: {}".format(scan_dimension), file=output)
    print(headprinter(), file=output)

    for x in xr:
        i = scan_dimension
        xx[i] = x
        elec = elec.update(xx, last_elec)
        print(lineprinter(xx, model, elec), file=output)

        last_elec = elec


if __name__ == "__main__":
    main()
