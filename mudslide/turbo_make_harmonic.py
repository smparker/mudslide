# -*- coding: utf-8 -*-
"""
Extract harmonic parameters from a vibrational analysis
"""

from __future__ import annotations

from typing import Any
import argparse
import sys

import numpy as np

from .models.turbomole_model import TurboControl, turbomole_is_installed_or_prefixed
from .models.harmonic_model import HarmonicModel
from .units import amu
from .version import get_version_info



def add_make_harmonic_parser(subparsers: Any) -> None:
    """Add make_harmonic subparser to an argument parser.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        Subparser object to add the make_harmonic parser to

    Returns
    -------
    None
        Modifies subparsers in place by adding a make_harmonic subparser
    """
    parser = subparsers.add_parser(
        "make_harmonic",
        help="Generate a harmonic model from a vibrational analysis")
    add_make_harmonic_arguments(parser)
    parser.set_defaults(func=make_harmonic_main)



def add_make_harmonic_arguments(parser: Any) -> None:
    """Add command line arguments for make_harmonic command.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to add arguments to

    Returns
    -------
    None
        Modifies parser in place by adding arguments
    """
    parser.add_argument("-c",
                        "--control",
                        default="control",
                        help="Control file")
    parser.add_argument("-d",
                        "--model-dest",
                        default="harmonic.json",
                        help="Where to write harmonic model as a json output")
    parser.add_argument("-o",
                        "--output",
                        default=sys.stdout,
                        type=argparse.FileType('w'),
                        help="Where to print output")



def main(argv: list[str] | None = None) -> None:
    """Parse command line arguments and run make_harmonic command.

    Parameters
    ----------
    argv : list of str, optional
        Command line arguments. If None, uses sys.argv[1:], by default None

    Returns
    -------
    argparse.Namespace
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Generate a harmonic model from a vibrational analysis",
        epilog=get_version_info(),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v',
                        '--version',
                        action='version',
                        version=get_version_info())
    add_make_harmonic_arguments(parser)
    args = parser.parse_args(argv)

    make_harmonic_wrapper(args)



def make_harmonic_wrapper(args: Any) -> None:
    """Wrapper function for make_harmonic command.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments

    Returns
    -------
    None
        Executes the make_harmonic command
    """
    make_harmonic_main(args.control, args.model_dest, args.output)



def make_harmonic_main(control: str, model_dest: str, output: Any) -> None:
    """Main function for make_harmonic command.

    Parameters
    ----------
    control : str
        Path to Turbomole control file
    model_dest : str
        Path to write harmonic model
    output : Any
        Output stream

    Returns
    -------
    None
        Writes harmonic model to model_dest
    """
    if not turbomole_is_installed_or_prefixed():
        raise RuntimeError("Turbomole is not available")

    print(f"Reading Turbomole control file from {control}", file=output)
    print(file=output)

    # get Turbomole control file
    turbo = TurboControl(control)

    # read coords
    symbols, coords = turbo.read_coords()
    masses = turbo.get_masses(symbols)

    print("Reference geometry:", file=output)
    print(
        f"{'el':>3s} {'x (Å)':>20s} {'y (Å)':>20s} {'z (Å)':>20s} {'mass (amu)':>20s}",
        file=output)
    print("-" * 100, file=output)

    ms = masses.reshape(-1, 3)[:, 0]
    for symbol, coord, mass in zip(symbols, coords.reshape(-1, 3), ms):
        print(
            f"{symbol:3s} "
            f"{coord[0]: 20.16g} {coord[1]: 20.16g} {coord[2]: 20.16g} "
            f"{mass / amu: 20.16g}",
            file=output)
    print(file=output)

    # read Hessian
    hessian = turbo.read_hessian()
    print("Hessian loaded with eigenvalues (H)", file=output)
    vals = np.linalg.eigvalsh(hessian)
    for i, val in enumerate(sorted(vals, key=abs)):
        print(f"  {i:3d} {val:20.10g}", file=output)
    print(file=output)

    harmonic = HarmonicModel(coords,
                             0.0,
                             hessian,
                             masses,
                             symbols,
                             ndims=3,
                             nparticles=len(symbols))

    print(f"Writing harmonic model to {model_dest}", file=output)
    harmonic.to_file(model_dest)
