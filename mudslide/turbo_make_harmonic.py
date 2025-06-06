# -*- coding: utf-8 -*-
"""
Extract harmonic parameters from a vibrational analysis
"""

from typing import Any
import argparse
import sys

import numpy as np

from mudslide.models.turbomole_model import TurboControl, turbomole_is_installed

def add_make_harmonic_parser(subparsers):
    parser = subparsers.add_parser("make_harmonic", help="Generate a harmonic model from a vibrational analysis")
    add_make_harmonic_arguments(parser)
    parser.set_defaults(func=make_harmonic_main)

def add_make_harmonic_arguments(parser):
    parser.add_argument("-c", "--control", default="control", help="Control file")
    parser.add_argument("-d", "--model-dest", default="harmonic.json", help="Where to write harmonic model as a json output")
    parser.add_argument("-o", "--output", default=sys.stdout, type=argparse.FileType('w'),
                        help="Where to print output")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate a harmonic model from a vibrational analysis")
    add_make_harmonic_arguments(parser)
    args = parser.parse_args(argv)

def make_harmonic_wrapper(args):
    make_harmonic_main(args.control, args.model_dest, args.output)

def make_harmonic_main(control: str, model_dest: str, output: Any):
    if not turbomole_is_installed():
        raise RuntimeError("Turbomole is not available")

    print(f"Reading Turbomole control file from {control}", file=output)
    print(file=output)

    # get Turbomole control file
    turbo = TurboControl(control)

    # read coords
    symbols, coords = turbo.read_coords()
    masses = turbo.get_masses(symbols)

    print("Reference geometry:", file=output)
    print(f"{'el':>3s} {'x':>20s} {'y':>20s} {'z':>20s} {'mass':>20s}", file=output)
    print("-" * 100, file=output)

    ms = masses.reshape(-1, 3)[:, 0]
    for symbol, coord, mass in zip(symbols, coords.reshape(-1, 3), ms):
        print(f"{symbol:3s} {coord[0]: 20.16g} {coord[1]: 20.16g} {coord[2]: 20.16g} {mass: 20.16g})", file=output)
    print(file=output)

    # read Hessian
    hessian = turbo.read_hessian()
    print("Hessian loaded with eigenvalues", file=output)
    print(np.linalg.eigvalsh(hessian), file=output)
    print(file=output)

    harmonic = HarmonicModel(coords, 0.0, hessian, masses, symbols, ndims=3, nparticles=len(symbols))

    print(f"Writing harmonic model to {model_dest}", file=output)
    harmonic.to_file(model_dest)
